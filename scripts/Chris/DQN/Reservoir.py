import numpy as np
import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDP


class Reservoir(Network):
  def __init__(self,
               in_size,   # Number of input neurons
               exc_size,  # Number of excitatory neurons
               inh_size,  # Number of inhibitory neurons
               w_in_exc,  # Input to excitatory weights
               w_in_inh,  # Input to inhibitory weights
               w_exc_exc, # Excitatory to excitatory weights
               w_exc_inh, # Excitatory to inhibitory weights
               w_inh_exc, # Inhibitory to excitatory weights
               w_inh_inh, # Inhibitory to inhibitory weights
               hyper_params,  # Dictionary of hyperparameters
               device='cpu'):
    super().__init__()

    ## Layers ##
    input = Input(n=in_size)
    res_exc = AdaptiveLIFNodes(
      n=exc_size,
      thresh=hyper_params['exc_thresh'],
      theta_plus=hyper_params['exc_theta_plus'],
      refrac=hyper_params['exc_refrac'],
      reset=hyper_params['exc_reset'],
      tc_theta_decay=hyper_params['exc_tc_theta_decay'],
      tc_decay=hyper_params['exc_tc_decay'],
      traces=True,
    )
    exc_monitor = Monitor(res_exc, ["s"], device=device)
    self.add_monitor(exc_monitor, name='res_monitor_exc')
    self.exc_monitor = exc_monitor
    res_inh = AdaptiveLIFNodes(
      n=inh_size,
      thresh=hyper_params['inh_thresh'],
      theta_plus=hyper_params['inh_theta_plus'],
      refrac=hyper_params['inh_refrac'],
      reset=hyper_params['inh_reset'],
      tc_theta_decay=hyper_params['inh_tc_theta_decay'],
      tc_decay=hyper_params['inh_tc_decay'],
      traces=True,
    )
    inh_monitor = Monitor(res_inh, ["s"], device=device)
    self.add_monitor(inh_monitor, name='res_monitor_inh')
    self.inh_monitor = inh_monitor
    self.add_layer(input, name='input')
    self.add_layer(res_exc, name='res_exc')
    self.add_layer(res_inh, name='res_inh')

    ## Connections ##
    in_exc_wfeat = Weight(name='in_exc_weight_feature', value=torch.Tensor(w_in_exc),)
    in_exc_conn = MulticompartmentConnection(
      source=input, target=res_exc,
      device=device, pipeline=[in_exc_wfeat],
    )
    in_inh_wfeat = Weight(name='in_inh_weight_feature', value=torch.Tensor(w_in_inh),)
    in_inh_conn = MulticompartmentConnection(
      source=input, target=res_inh,
      device=device, pipeline=[in_inh_wfeat],
    )
    exc_exc_wfeat = Weight(name='exc_exc_weight_feature', value=torch.Tensor(w_exc_exc),)
    exc_exc_conn = MulticompartmentConnection(
      source=res_exc, target=res_exc,
      device=device, pipeline=[exc_exc_wfeat],
    )
    exc_inh_wfeat = Weight(name='exc_inh_weight_feature', value=torch.Tensor(w_exc_inh),)
    exc_inh_conn = MulticompartmentConnection(
      source=res_exc, target=res_inh,
      device=device, pipeline=[exc_inh_wfeat],
    )
    inh_exc_wfeat = Weight(name='inh_exc_weight_feature', value=torch.Tensor(w_inh_exc),)
    inh_exc_conn = MulticompartmentConnection(
      source=res_inh, target=res_exc,
      device=device, pipeline=[inh_exc_wfeat],
    )
    inh_inh_wfeat = Weight(name='inh_inh_weight_feature', value=torch.Tensor(w_inh_inh),)
    inh_inh_conn = MulticompartmentConnection(
      source=res_inh, target=res_inh,
      device=device, pipeline=[inh_inh_wfeat],
    )
    self.add_connection(in_exc_conn, source='input', target='res_exc')
    self.add_connection(in_inh_conn, source='input', target='res_inh')
    self.add_connection(exc_exc_conn, source='res_exc', target='res_exc')
    self.add_connection(exc_inh_conn, source='res_exc', target='res_inh')
    self.add_connection(inh_exc_conn, source='res_inh', target='res_exc')
    self.add_connection(inh_inh_conn, source='res_inh', target='res_inh')

    ## Migrate ##
    self.to(device)


  # Expect input_train to be an ndarray of shape (#-Grid-Cells, time)
  def get_spikes(self, input_train: np.ndarray, sim_time):
    input_train = torch.Tensor(input_train.T).unsqueeze(1)    # Reshape to (time, 1, #-Grid-Cells)
    self.run(inputs={'input': input_train}, time=sim_time,)
    exc_spikes = self.exc_monitor.get('s')
    inh_spikes = self.inh_monitor.get('s')
    return exc_spikes, inh_spikes
