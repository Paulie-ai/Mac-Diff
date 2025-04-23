import torch
from tqdm import tqdm


def extract_to_tensor(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionSampler(torch.nn.Module):

    def __init__(self,
                 model,
                 timesteps=1000,
                 betas=None,
                 beta_schedule='linear',
                 loss_type='l2',
                 linear_start=0.01,
                 linear_end=0.2,
                 cosine_s=8e-3,
                 sampling_steps=1000,
                 ddim_eta=1.,
                 w=0.,
                 ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.sampling_steps = sampling_steps
        self.ddim_eta = ddim_eta
        self.w = w

        self.register_schedule(betas=betas,
                               beta_schedule=beta_schedule,
                               timesteps=timesteps,
                               linear_start=linear_start,
                               linear_end=linear_end,
                               cosine_s=cosine_s)

        # ----------End of __init__--------------

    def register_schedule(self, betas, beta_schedule, timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        if betas is None:
            betas = torch.linspace(linear_start, linear_end, timesteps).to('cuda')

        alphas = 1. - betas

        # cumulative product of alphas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones_like(alphas[:1]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculating q(x_t | x_{t-1}), add noise to x_{t-1}
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.rsqrt(alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1.))


    @torch.no_grad()
    def denoise_sample_from_pure_noise(self, shape, cond=None, return_intermediates=False):
        # use ddim sampling
        return self.ddim_sample(shape, cond)

    def ddim_sample(self, shape, context, clip_scheme='static'):
        '''

        :param shape: input shape
        :param context: conditioning, here our textual description embedding
        :param clip_scheme: 'static' or 'dynamic'
        :return: a less noisy sample
        '''

        batch_size = shape[0]
        device = self.betas.device
        total_timesteps = self.timesteps
        ddim_steps = self.sampling_steps
        eta = self.ddim_eta

        times = torch.linspace(-1, total_timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        output = torch.randn(shape, device=device)
        x_start = None

        for t, t_next in tqdm(time_pairs, desc='DDIM sampling loop time step'):
            time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(output, time_cond, context, clip_scheme=clip_scheme)

            if t_next < 0:
                output = x_start
                continue

            alpha_bar = self.alphas_cumprod[t]
            alpha_next_bar = self.alphas_cumprod[t_next]

            sigma = torch.sqrt(eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
            c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

            noise = torch.randn_like(output)

            output = x_start * torch.sqrt(alpha_next_bar) + \
                     c * pred_noise + \
                     sigma * noise

        return output

    def model_predictions(self, x, time_cond, context, clip_scheme='static'):
        '''
        :param x: input image
        :param cond: conditioning, here our textual description embedding
        :param time_cond: time step
        :param clip_scheme: 'static' or 'dynamic'
        :return: prediction noise, x_start
        '''

        if context is None:
            model_output = self.model(x, time_cond, context)
        else:
            model_output = self.w * self.model(x, time_cond, context) + (1 - self.w) * self.model(x, time_cond, context * 0)

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, time_cond, pred_noise)
        if clip_scheme == 'static':
            x_start = torch.clamp(x_start, -1., 1.)
        elif clip_scheme == 'dynamic':
            # TODO: see Imagen implementation
            pass

        return pred_noise, x_start

    def predict_start_from_noise(self, x_t, t, pred_noise):
        return extract_to_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               extract_to_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract_to_tensor(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start + \
               extract_to_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_loss(self, x_start, t, context, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = self.model(x_noisy, t, context)

        if self.loss_type == 'l1':
            loss = torch.abs(pred_noise - noise).mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
        else:
            raise NotImplementedError(f'Loss type "{self.loss_type}" not implemented')

        return loss

    def forward(self, x, context=None):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        return self.p_loss(x, t, context)

