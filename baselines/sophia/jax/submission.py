"""Submission file for an NAdamW optimizer with warmup+cosine LR in Jax."""

import functools

# isort: off
# We have to turn off isort here to resolve a conflict between isort and yapf.
from typing import (Any,
                    Callable,
                    Dict,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)
# isort: on

import chex
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec

_GRAD_CLIP_EPS = 1e-6

HPARAMS = {
    "learning_rate": 1e-4,
    "b1": 0.965,
    "b2": 0.99,
    "rho": 0.04,
    "weight_decay": 1e-1,
    "k": 100,
    "warmup_factor": 0.02
}


def sophia(
    learning_rate: Union[float, optax.Schedule],
    b1: float = 0.965,
    b2: float = 0.99,
    rho: float = 0.04,
    weight_decay: float = 1e-1,
    k = 10,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  
  return optax.chain(
      scale_by_sophia(b1, b2, rho, k),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      scale_by_learning_rate(learning_rate))

def scale_by_sophia(b1: float = 0.965,
                   b2: float = 0.99,
                   rho: float = 0.04,
                   k = 10) -> optax.GradientTransformation:

  def init_fn(params):
    m = jax.tree_map(jnp.zeros_like, params)
    g = jax.tree_map(jnp.zeros_like, params)
    h = jax.tree_map(jnp.zeros_like, params)
    count = jnp.array(0, dtype=jnp.int32)
    return ScaleBySophiaState(count=count, m=m, g=g, h=h)


  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    if state.count % k == 0:
      state.h = _update_hess(updates, state.h, b2, state.count)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleBySophiaState(NamedTuple):
  """State for the Sophia algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: optax.Updates
  g: optax.Updates
  h: optax.Updates
  


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

def _update_hess(updates, moments, decay, count):
  if count == 0:
    pass
  else:
    return jax.tree_map(
        lambda g, t: (1 - decay) * jnp.square(g) + decay * t, updates, moments)
  

def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Initializes Sophia optimizer state."""
  del model_params
  del model_state
  del rng
  del hyperparameters

  hyperparameters = HPARAMS
  print(hyperparameters)
  def jax_cosine_warmup(step_hint: int, hyperparameters):
    # Create learning rate schedule.
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters.learning_rate,
        transition_steps=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])
    return schedule_fn

  # Create optimizer + LR schedule.
  # lr_schedule_fn = jax_cosine_warmup(workload.step_hint * 0.75, hyperparameters)

  opt_init_fn, opt_update_fn = sophia(
    learning_rate=1e-4,
    b1=hyperparameters.b1,
    b2=hyperparameters.b2,
    rho = hyperparameters.rho,
    weight_decay=hyperparameters.weight_decay,
    k = hyperparameters.k)
    
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn

@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip):
    def _loss_fn(params):
        """Loss function used for training."""
        logits, new_model_state = workload.model_fn(
            params,
            batch,
            model_state,
            spec.ForwardPassMode.TRAIN,
            rng,
            update_batch_norm=True)
        loss_dict = workload.loss_fn(
            label_batch=batch['targets'],
            logits_batch=logits,
            mask_batch=batch.get('weights'))
        summed_loss = loss_dict['summed']
        n_valid_examples = loss_dict['n_valid_examples']
        return summed_loss, (n_valid_examples, new_model_state)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
        current_param_container)
    # Get correct global mean loss and grad.
    (summed_loss, n_valid_examples, grad) = lax.psum(
        (summed_loss, n_valid_examples, grad), axis_name='batch')
    loss = summed_loss / n_valid_examples
    grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

    grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))
    
    if grad_clip is not None:
        grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
        grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
        grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

    updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
    updated_params = optax.apply_updates(current_param_container, updates)

    return new_optimizer_state, updated_params, new_model_state, loss, grad_norm

def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
    """Return (updated_optimizer_state, updated_params, updated_model_state)."""
    del current_params_types, loss_type, eval_results, hyperparameters

    hyperparameters = HPARAMS

    optimizer_state, opt_update_fn = optimizer_state
    per_device_rngs = jax.random.split(rng, jax.local_device_count())
    if hasattr(hyperparameters, 'label_smoothing'):
      label_smoothing = hyperparameters.label_smoothing
    else:
      label_smoothing = 0.0
    if hasattr(hyperparameters, 'grad_clip'):
      grad_clip = hyperparameters.grad_clip
    else:
      grad_clip = None
    outputs = pmapped_train_step(workload,
                                 opt_update_fn,
                                 model_state,
                                 optimizer_state,
                                 current_param_container,
                                 batch,
                                 per_device_rngs,
                                 grad_clip,
                                 label_smoothing)
    new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

    # Log loss, grad_norm (if applicable).
    if global_step % 100 == 0 and workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
            {'loss': loss[0], 'grad_norm': grad_norm[0]}, global_step)

    return (new_optimizer_state, opt_update_fn), new_params, new_model_state

def get_batch_size(workload_name):
    # Return the global batch size.
    if workload_name == 'criteo1tb':
      return 262_144
    elif workload_name == 'fastmri':
      return 32
    elif workload_name == 'imagenet_resnet':
      return 1024
    elif workload_name == 'imagenet_vit':
      return 1024
    elif workload_name == 'librispeech_conformer':
      return 256
    elif workload_name == 'librispeech_deepspeech':
      return 256
    elif workload_name == 'ogbg':
      return 512
    elif workload_name == 'wmt':
      return 128
    elif workload_name == 'mnist':
      return 16
    else:
      raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
