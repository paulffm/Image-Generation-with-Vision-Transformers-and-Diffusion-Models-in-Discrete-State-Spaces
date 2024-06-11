import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import lib.sampling.sampling_utils as sampling_utils
import lib.utils.utils as utils
import time
from functools import partial
from lib.models.model_utils import get_logprob_with_logits
import time


# Sampling observations:
def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == "uniform":
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == "gaussian":
        target = np.exp(
            -((np.arange(1, S + 1) - S // 2) ** 2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)

        cat = torch.distributions.categorical.Categorical(torch.from_numpy(target))
        x = cat.sample((N * D,)).view(N, D)
        x = x.to(device)
    else:
        raise NotImplementedError("Unrecognized initial dist " + initial_dist)
    return x


def get_reverse_rates(model, logits, x, t_ones, cfg, N, D, S):
    if cfg.loss.name == "CTElbo" or cfg.loss.name == "NLL":
        device = model.device
        qt0 = model.transition(t_ones)  # (N, S, S)
        rate = model.rate(t_ones)  # (N, S, S)

        p0t = F.softmax(logits, dim=2)  # (N, D, S) (not log_softmax)

        qt0_denom = (
            qt0[
                torch.arange(N, device=device).repeat_interleave(D * S),
                torch.arange(S, device=device).repeat(N * D),
                x.long().flatten().repeat_interleave(S),
            ].view(N, D, S)
            + cfg.sampler.eps_ratio
        )

        # First S is x0 second S is x tilde
        qt0_numer = qt0  # (N, S, S)

        forward_rates = rate[
            torch.arange(N, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(N * D),
            x.long().flatten().repeat_interleave(S),
        ].view(N, D, S)

        ratio = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)

        reverse_rates = forward_rates * ratio  # (N, D, S)

    elif cfg.loss.name == "CatRM" or cfg.loss.name == "CatRMNLL" or "ScoreElbo":
        ll_all, ll_xt = get_logprob_with_logits(
            cfg=cfg,
            model=model,
            xt=x,
            t=t_ones,
            logits=logits,
        )

        log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
        fwd_rate = model.rate_mat(x.long(), t_ones)  # B, D, S
        ratio = torch.exp(log_weight)
        reverse_rates = ratio * fwd_rate
    else:
        raise ValueError(f"No loss named {cfg.loss.name}")

        # B, D, S
    return reverse_rates, ratio


@sampling_utils.register_sampler
class TauL:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        self.is_ordinal = cfg.sampler.is_ordinal
        self.loss_name = cfg.loss.name

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_dim = []
            change_jumps = []
            change_mjumps = []
            # new_tensor = torch.empty((self.num_steps, N, self.D), device=device)

            time_list = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                start_time = time.time()
                # new_tensor[idx] = x
                h = ts[idx] - ts[idx + 1]
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                logits = model(x, t_ones)

                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                xt_onehot = F.one_hot(x.long(), self.S)
                reverse_rates = reverse_rates * (1 - xt_onehot)
                poisson_dist = torch.distributions.poisson.Poisson(
                    reverse_rates * h
                )  # posterior: p_{t-eps|t}, B, D; S
                jump_nums = (
                    poisson_dist.sample()
                )  # how many jumps in interval [t-eps, t]
                if not self.is_ordinal:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)

                    # proportion of jumps
                    # jump_num_sum = torch.sum(jump_nums, dim=2)
                    # wv dim changen
                    # change_dim.append(torch.mean(((jump_num_sum > 0) * 1).to(dtype=float)).item())
                    # in wv dim sind multiple jumps
                    # change_jumps.append(torch.mean(((jump_num_sum > 1) * 1).to(dtype=float)).item())
                    # wv prop von changes sind multiple changes
                    # changes = torch.sum(((jump_num_sum > 0) * 1).to(dtype=float))
                    # changes_rej = torch.sum(((jump_num_sum > 1) * 1).to(dtype=float))
                    # change_mjumps.append((changes_rej/changes).item())

                choices = utils.expand_dims(
                    torch.arange(self.S, device=device, dtype=torch.int32),
                    axis=list(range(x.ndim)),
                )
                diff = choices - x.unsqueeze(-1)
                adj_diffs = jump_nums * diff
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump

                x_new = torch.clamp(xp, min=0, max=self.S - 1)
                changes = torch.sum(x != x_new)
                change_dim.append(changes.cpu().numpy() / N)

                x = x_new
                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)

                        t_h_ones = (t) * torch.ones((N,), device=device)
                        rate = model.rate(t_h_ones)

                        logits = model(x_new, t_h_ones)  #
                        reverse_rates, _ = get_reverse_rates(
                            model, logits, x_new, t_h_ones, self.cfg, N, self.D, self.S
                        )
                        reverse_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0

                        transpose_forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            x_new.long().flatten().repeat_interleave(self.S),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                        ].view(N, self.D, self.S)

                        corrector_rates = (
                            transpose_forward_rates + reverse_rates
                        )  # (N, D, S)
                        corrector_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0
                        poisson_dist = torch.distributions.poisson.Poisson(
                            corrector_rates * h
                        )  # posterior: p_{t-eps|t}, B, D; S

                        jump_nums = (
                            poisson_dist.sample()
                        )  # how many jumps in interval [t-eps, t]

                        if not self.is_ordinal:
                            jump_num_sum = torch.sum(jump_nums, dim=2)
                            jump_num_sum_mask = jump_num_sum <= 1
                            jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
                        choices = utils.expand_dims(
                            torch.arange(self.S, device=device, dtype=torch.int32),
                            axis=list(range(x_new.ndim)),
                        )
                        diff = choices - x_new.unsqueeze(-1)
                        adj_diffs = jump_nums * diff
                        overall_jump = torch.sum(adj_diffs, dim=2)
                        xp = x_new + overall_jump

                        x_new = torch.clamp(xp, min=0, max=self.S - 1)
                        x = x_new

            if self.loss_name == "CTElbo" or self.loss_name == "NLL":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x

            return (
                x_0max.detach().cpu().numpy().astype(int),  # change_jump,
                change_dim,
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class LBJF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        self.loss_name = cfg.loss.name

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_dim = []
            new_tensor = torch.empty((self.num_steps, N, self.D), device=device)
            for idx, t in tqdm(enumerate(ts[0:-1])):
                new_tensor[idx] = x
                h = ts[idx] - ts[idx + 1]
                t_ones = t * torch.ones((N,), device=device)
                qt0 = model.transition(t_ones)  # (N, S, S)
                rate = model.rate(t_ones)
                logits = model(x, t_ones)
                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                xt_onehot = F.one_hot(x.long(), self.S)
                post_0 = reverse_rates * (1 - xt_onehot)

                off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                reverse_rates = post_0 * h + diag * xt_onehot  # * h  # eq.17

                reverse_rates = reverse_rates / torch.sum(
                    reverse_rates, axis=-1, keepdims=True
                )
                log_posterior = torch.log(reverse_rates + 1e-35).view(-1, self.S)
                x_new = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )
                changes = torch.sum(x != x_new)
                change_dim.append(changes.cpu().numpy() / N)
                if t <= self.corrector_entry_time:
                    for _ in range(self.num_corrector_steps):
                        print("corrector")
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        t_h_ones = t * torch.ones((N,), device=device)
                        logits = model(x_new, t_h_ones)  #
                        reverse_rates, _ = get_reverse_rates(
                            model, logits, x_new, t_h_ones, self.cfg, N, self.D, self.S
                        )

                        transpose_forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            x_new.long().flatten().repeat_interleave(self.S),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                        ].view(N, self.D, self.S)

                        corrector_rates = (
                            transpose_forward_rates + reverse_rates
                        )  # (N, D, S)
                        corrector_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0
                        xt_new_onehot = F.one_hot(x_new.long(), self.S)
                        post_0 = corrector_rates * (1 - xt_new_onehot)

                        off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                        diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                        corrector_rates = post_0 * h + diag * xt_new_onehot
                        corrector_rates = corrector_rates / torch.sum(
                            corrector_rates, axis=-1, keepdims=True
                        )
                        log_posterior = torch.log(corrector_rates + 1e-35).view(
                            -1, self.S
                        )

                        x_new = (
                            torch.distributions.categorical.Categorical(
                                logits=log_posterior
                            )
                            .sample()
                            .view(N, self.D)
                        )
                # print(torch.sum(x_new != x, dim=1))

                x = x_new
            if self.loss_name == "CTElbo" or self.loss_name == "NLL":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_dim,
                # new_tensor.detach().cpu().numpy().astype(int),
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class MidPointTauL:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.is_ordinal = cfg.sampler.is_ordinal
        self.device = cfg.device
        self.eps_ratio = cfg.sampler.eps_ratio
        self.loss_name = cfg.loss.name

        if cfg.data.name == "DiscreteMNIST":
            self.state_change = -torch.load(
                "SavedModels/MNIST/state_change_matrix_mnist.pth"
            )
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "Maze3S":
            self.state_change = -torch.tensor(
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], device=self.device
            )
            # self.state_change = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], device=self.device)
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "BinMNIST" or cfg.data.name == "SyntheticData":
            self.state_change = -torch.tensor([[0, 1], [-1, 0]], device=self.device)

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        self.state_change = torch.tile(self.state_change, (N, 1, 1))
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )

            t = self.max_t
            change_jump = []
            change_clamp1 = []
            change_clamp2 = []
            change_dim = []
            change_dim_first = []
            change_1to2 = []
            h = (self.max_t - self.min_t) / self.num_steps
            # Fragen:
            # 1. Prediction zum  Zeitpunkt 0.5 * h +t_ones?
            # Wie summe über states? => meistens R * changes = 0
            #
            i = 0
            count = 0
            while t - 0.5 * h > self.min_t:
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                t_05 = t_ones - 0.5 * h
                logits = model(x, t_ones)

                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0

                # print(t, reverse_rates)
                # achtung ein verfahren definieren mit:
                # x_prime und eins mit echtem

                state_change = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                xt_onehot = F.one_hot(x.long(), self.S)

                change = torch.round(
                    0.5 * h * torch.sum((reverse_rates * state_change), dim=-1)
                ).to(dtype=torch.int)
                xp_prime = x + change  # , dim=-1)

                """
                if (change == torch.zeros((N, self.D), device=self.device)).all():
                    print("True")
                else:
                    count = count + 1
                    print("False")
                """
                x_prime = torch.clip(xp_prime, min=0, max=self.S - 1)

                # change_clamp1.append((torch.sum(xp_prime != x_prime) / (N * self.D)).item())
                change_dim_first.append((torch.sum(x != x_prime) / (N * self.D)).item())

                # ------------second-------------------
                logits_prime = model(x_prime, t_05)

                reverse_rates_prime, _ = get_reverse_rates(
                    model, logits_prime, x_prime, t_05, self.cfg, N, self.D, self.S
                )

                reverse_rates_prime[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x_prime.long().flatten(),
                ] = 0.0

                state_change_prime = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x_prime.long()
                    .flatten()
                    .repeat_interleave(self.S),  # wenn hier x_prime
                ].view(N, self.D, self.S)

                diff_prime = state_change_prime

                flips = torch.distributions.poisson.Poisson(
                    reverse_rates_prime * h
                ).sample()  # B, D most 0

                if not self.is_ordinal:
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1
                    flips = flips * flip_mask
                else:
                    jump_num_sum = torch.sum(flips, axis=-1)
                    changes = torch.sum(((jump_num_sum > 0) * 1).to(dtype=float))
                    # print("1", changes)
                    changes_rej = torch.sum(((jump_num_sum > 1) * 1).to(dtype=float))
                    # proportion of jumps thate are multiple jumps
                    change_jump.append((changes_rej / changes).item())
                # diff = choices - x.unsqueeze(-1)

                avg_offset = torch.sum(
                    flips * diff_prime, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1
                xp = x + avg_offset  # wenn hier x_prime

                x_new = torch.clip(xp, min=0, max=self.S - 1)
                # change_clamp2.append((torch.sum(xp != x) / (N * self.D)).item())
                change_dim.append((torch.sum(xp != x) / (N * self.D)).item())
                change_1to2.append((torch.sum(x_prime != x_new) / (N * self.D)).item())

                x = x_new
                t = t - h
                i += 1

            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x

            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_jump,
                change_dim,
                change_dim_first,
                change_1to2,
            )  # change_clamp1, change_clamp2


@sampling_utils.register_sampler
class PCTauL:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N):
        t = 1.0
        D = self.cfg.model.concat_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        device = model.device

        initial_dist = scfg.initial_dist
        initial_dist_std = 200  # model.Q_sigma

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist, initial_dist_std)

            h = 1.0 / num_steps  # approximately
            ts = np.linspace(1.0, min_t + h, num_steps)

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)
                    rate = model.rate(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)

                    p0t = F.softmax(
                        model(in_x, in_t * torch.ones((N,), device=device)), dim=2
                    )  # (N, D, S)

                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = (
                        qt0[
                            torch.arange(N, device=device).repeat_interleave(D * S),
                            torch.arange(S, device=device).repeat(N * D),
                            in_x.long().flatten().repeat_interleave(S),
                        ].view(N, D, S)
                        + eps_ratio
                    )

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0  # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D * S),
                        torch.arange(S, device=device).repeat(N * D),
                        in_x.long().flatten().repeat_interleave(S),
                    ].view(N, D, S)

                    reverse_rates = forward_rates * (
                        (p0t / qt0_denom) @ qt0_numer
                    )  # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten(),
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D * S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N * D),
                    ].view(N, D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1, 1, S) - in_x.view(
                        N, D, 1
                    )
                    poisson_dist = torch.distributions.poisson.Poisson(
                        in_reverse_rates * in_h
                    )
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S - 1)

                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t - h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten(),
                        ] = 0.0
                        x = take_poisson_step(
                            x, corrector_rate, corrector_step_size_multiplier * h
                        )

            p_0gt = F.softmax(
                model(x, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int)  # , x_hist, x0_hist


@sampling_utils.register_sampler
class ConditionalTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        reject_multiple_jumps = scfg.reject_multiple_jumps
        initial_dist = scfg.initial_dist
        if initial_dist == "gaussian":
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, sample_D, device, S, initial_dist, initial_dist_std
            )

            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(
                    model(model_input, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                p0t = p0t[:, condition_dim:, :]

                x_0max = torch.max(p0t, dim=2)[1]

                qt0_denom = (
                    qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D * S),
                        torch.arange(S, device=device).repeat(N * sample_D),
                        x.long().flatten().repeat_interleave(S),
                    ].view(N, sample_D, S)
                    + eps_ratio
                )

                # First S is x0 second S is x tilde

                qt0_numer = qt0  # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(sample_D * S),
                    torch.arange(S, device=device).repeat(N * sample_D),
                    x.long().flatten().repeat_interleave(S),
                ].view(N, sample_D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)

                reverse_rates = forward_rates * inner_sum  # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(sample_D),
                    torch.arange(sample_D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1, 1, S) - x.view(
                    N, sample_D, 1
                )
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()

                if reject_multiple_jumps:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    masked_jump_nums = jump_nums * jump_num_sum_mask.view(
                        N, sample_D, 1
                    )
                    adj_diffs = masked_jump_nums * diffs
                else:
                    adj_diffs = jump_nums * diffs

                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S - 1)

                x = x_new

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(
                model(model_input, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int)


@sampling_utils.register_sampler
class ConditionalPCTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == "gaussian":
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, sample_D, device, S, initial_dist, initial_dist_std
            )

            h = 1.0 / num_steps  # approximately
            ts = np.linspace(1.0, min_t + h, num_steps)

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)
                    rate = model.rate(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(
                        model(model_input, in_t * torch.ones((N,), device=device)),
                        dim=2,
                    )  # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]

                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = (
                        qt0[
                            torch.arange(N, device=device).repeat_interleave(
                                sample_D * S
                            ),
                            torch.arange(S, device=device).repeat(N * sample_D),
                            x.long().flatten().repeat_interleave(S),
                        ].view(N, sample_D, S)
                        + eps_ratio
                    )

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0  # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D * S),
                        torch.arange(S, device=device).repeat(N * sample_D),
                        in_x.long().flatten().repeat_interleave(S),
                    ].view(N, sample_D, S)

                    reverse_rates = forward_rates * (
                        (p0t / qt0_denom) @ qt0_numer
                    )  # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten(),
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D * S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N * sample_D),
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1, 1, S) - in_x.view(
                        N, sample_D, 1
                    )
                    poisson_dist = torch.distributions.poisson.Poisson(
                        in_reverse_rates * in_h
                    )
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(
                            N, sample_D, 1
                        )
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S - 1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t - h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten(),
                        ] = 0.0
                        x = take_poisson_step(
                            x, corrector_rate, corrector_step_size_multiplier * h
                        )

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(
                model(model_input, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int)
