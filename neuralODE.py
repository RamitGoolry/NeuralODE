import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.autograd as autograd

from torch import Tensor

from ODESolve import runge_kutta4, euler

import numpy as np

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        batch_size = z.shape[0]

        out = self.forward(z, t)  # f(z, t)

        a = grad_outputs

        a_df_dz, a_df_dt, *a_df_dp = autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )

        if a_df_dp is not None:
            a_df_dp = torch.cat([p_grad.flatten() for p_grad in a_df_dp]).unsqueeze(0)
            a_df_dp = a_df_dp.expand(batch_size, -1) / batch_size
        if a_df_dt is not None:
            a_df_dt = a_df_dt.expand(batch_size, -1) / batch_size

        return out, a_df_dz, a_df_dt, a_df_dp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []

        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())

        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func, mode = 'rk4'):
        assert isinstance(func, ODEF)

        ode_solver = {
            'rk4' : runge_kutta4,
            'euler' : euler
        }[mode]

        bs, *z_shape = z0.size
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0

            for i_t in range(time_len - 1):
                z = ode_solver(z0, t[i_t], t[i_t + 1], func)
                z[i_t + 1] = z

        ctx.func = func
        ctx.solver = ode_solver
        ctx.save_for_backward(t, z.clone(), flat_parameters)

        return z

    @staticmethod
    def backward(ctx, dLdz):
        '''
        dLdz: (time_len, batch_size, *z_shape)
        '''

        func = ctx.func
        ode_solver = ctx.solver
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()

        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            '''
            Tensors are temporal slices
            t_i - is tensor with size: (bs, 1)
            aug_z_i is tensor with size: (bs, n_dim * 2 + n_params + 1)
            '''
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:n_dim*2]

            # unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs,*z_shape)

            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)

                func_eval, a_df_dz, a_df_dt, a_df_dp = func.forward_with_grad(z_i, t_i, grad_outputs=a)

                if a_df_dz is not None:
                    a_df_dz = a_df_dz.to(z_i)
                else:
                    a_df_dz = torch.zeros(bs, *z_shape).to(z_i)

                if a_df_dp is not None:
                    a_df_dp = a_df_dp.to(z_i)
                else:
                    a_df_dp = torch.zeros(bs, n_params).to(z_i)

                if a_df_dt is not None:
                    a_df_dt = a_df_dt.to(z_i)
                else:
                    a_df_dt = torch.zeros(bs, 1).to(z_i)

            # Flatten f and a_df_dz
            func_eval = func_eval.view(bs, n_dim)
            a_df_dz = a_df_dz.view(bs, n_dim)

            return torch.cat((func_eval, -a_df_dz, -a_df_dp, -a_df_dt), dim = 1)

        # Backward pass
        dLdz = dLdz.view(time_len, bs, n_dim)

        with torch.no_grad():
            # Placeholders for output gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)

            # We need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len - 1, 0, -1):
                # Calculate adjoint dynamics
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] -= dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solver(aug_z, t_i, t[i_t - 1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:n_dim*2]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim+n_params]
                adj_t[i_t - 1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans
            
            # Adjust time 0 adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0] #type: ignore

            # Adjust Adjoints
            adj_z += dLdz_0
            adj_t[0] -= dLdt_0
        
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)

        if return_whole_sequence:
            return z
        else:
            return z[-1]
