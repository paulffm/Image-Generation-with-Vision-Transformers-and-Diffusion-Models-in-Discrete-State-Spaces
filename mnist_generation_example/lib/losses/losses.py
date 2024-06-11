import torch
import torch.nn as nn
import lib.losses.losses_utils as losses_utils
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import lib.utils.utils as utils
import time
from lib.models.model_utils import get_logprob_with_logits

@losses_utils.register_loss
class CTElbo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.max_t = cfg.training.max_t
        self.cross_ent = nn.CrossEntropyLoss()

    def calc_loss(self, state, minibatch, label=None):
        model = state["model"]
        S = self.cfg.data.S
        # if 4 Dim => like images: True
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)

        B, D = minibatch.shape
        device = model.device

        # get random timestep between 1.0 and self.min_time
        ts = (
            torch.rand((B,), device=device) * (self.max_t - self.min_time)
            + self.min_time
        )  # 0.99999

        qt0 = model.transition(ts)

        # R_t = beta_t * R_b
        rate = model.rate(ts)  # (diagonal = - sum of rows)

        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(
                D
            ),  # repeats every element 0 to B-1 D-times
            minibatch.flatten().long(),  # minibatch.flatten() => (B, D) => (B*D) (1D-Tensor)
            :,
        ]  # (B*D, S)

        # set of (B*D) categorical distributions with probabilities from qt0_rows_reg
        log_qt0 = torch.where(qt0_rows_reg <= 0.0, -1e9, torch.log(qt0_rows_reg))
        x_t_cat = torch.distributions.categorical.Categorical(logits=log_qt0)
        x_t = x_t_cat.sample().view(  # sampling B * D times => from every row of qt0_rows_reg once => then transform it to shape B, D
            B, D
        )  # (B*D,) mit view => (B, D) Bsp: x_t = (0, 1, 2, 4, 3) (for B =1 )

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(D), x_t.long().flatten(), :
        ]  # (B*D, S)

        rate_vals_square[
            torch.arange(B * D, device=device), x_t.long().flatten()
        ] = 0.0  # 0 the diagonals

        rate_vals_square = rate_vals_square.view(B, D, S)  # (B*D, S) => (B, D, S)

        #  Summe der Werte entlang der Dimension S
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(
            B, D
        )  # B, D with every entry = S-1? => for entries of x_t same prob to transition?

        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )

        # Samples where transitions takes place in every row of B
        square_dims = square_dimcat.sample()  # (B,) taking values in [0, D)

        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device), square_dims, :
        ]  # (B, S) => every row has only one entry = 0, everywhere else 1; chooses the row square_dim of rate_vals_square

        # samples from rate_new_val_probs and chooses state to transition to => more likely where entry is 1 instead of 0?
        log_rate_new_val_probs = torch.where(
            rate_new_val_probs <= 0.0, -1e9, torch.log(rate_new_val_probs)
        )
        square_newvalcat = torch.distributions.categorical.Categorical(
            logits=log_rate_new_val_probs
        )

        # Samples state, where we going
        square_newval_samples = (
            square_newvalcat.sample()
        )  # (B, ) taking values in [0, S)

        x_tilde = x_t.clone()
        x_tilde[torch.arange(B, device=device), square_dims] = square_newval_samples

        # Now, when we minimize LCT, we are sampling (x, x ̃) from the forward process and then maximizing
        # the assigned model probability for the pairing in the reverse direction, just as in LDT

        # ---------- First term of ELBO (regularization) ---------------

        if self.one_forward_pass:
            #x_logits = model(x_tilde, ts)  # (B, D, S)
            x_logits = model(x_t, ts)
            # ensures that positive
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
            reg_x = x_tilde
        else:
            # x_t = x from Paper
            x_logits = model(x_t, ts)  # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
            reg_x = x_t

        # same as 1-one_hot => diagonals to 0
        mask_reg = torch.ones((B, D, S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten(),
        ] = 0.0  # (B, D, S)

        # q_{t|0} (x ̃|x_0)
        qt0_numer_reg = qt0.view(B, S, S)

        # q_{t|0} (x|x_0)

        qt0_denom_reg = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                :,
                reg_x.long().flatten(),
            ].view(B, D, S)
            + self.ratio_eps
        )

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten(),
        ].view(B, D, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(
            1, 2
        )  # (B, D, S)

        # first term; exactly as in formula
        reg_term = torch.sum((p0t_reg / qt0_denom_reg) * reg_tmp, dim=(1, 2))

        # ----- second term of continuous ELBO (signal term) ------------

        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            p0t_sig = F.softmax(model(x_tilde, ts), dim=2)  # (B, D, S)

        # q_{t|0} (x_0|x ̃)
        # prob of going from state S to any other state, for all states S
        qt0_numer_sig = qt0.view(B, S, S)  # first S is x_0, second S is x

        # q_{t | 0} (x_0|x ̃)
        # prob of going from any state to state S, specified by x_tilde in dimension d
        qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                :,
                x_tilde.long().flatten(),
            ].view(B, D, S)
            + self.ratio_eps
        )

        # log(R^theta_t(x ̃,x)) = R_t(x,x ̃) * sum_x_0 (q_{t|0} (x_0|x ̃) / q_{t | 0} (x_0|x ̃)) * ptheta_{0|t}(x_0|x)
        # only ratio
        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        )  # (B, D, S)

        # Masking of dimension d
        x_tilde_mask = torch.ones((B, D, S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten(),
        ] = 0.0

        # same forward rate: going from any State to state S, specifie by x_tilde
        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(B * D),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, D, S)

        # probability of going from state S, specified by x to any other state in this dimension
        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D * S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * D),  # all states
        ].view(B, D, S)

        # probability of going from state S, specified by x,to state S', specified by x_tilde, in this dimension
        outer_qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                minibatch.long().flatten(),
                x_tilde.long().flatten(),  # states x_tilde
            ]
            + self.ratio_eps
        )  # (B, D)

        outer_sum_sig = torch.sum(
            x_tilde_mask
            * outer_rate_sig  # forward rate B, D, S
            * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B, D, 1))
            * inner_log_sig,  # nur ratio der reverse rate
            dim=(1, 2),
        )

        # now getting the 2nd term normalization
        # Sum of transition rates for each row in a batch => for one state s, sum to transition to another state s' and s included
        rate_row_sums = -rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B),
        ].view(B, S)

        # choose transition rates for x_tilde
        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten(),
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp  # (B,D)
        Z_addition = rate_row_sums

        # Z_t => sum of R_t for all transition from x^d to tilde(x)^d but not x^d = tilde(x)^d => jumps 
        Z_sig_norm = (
            base_Z.view(B, 1, 1)
            - Z_subtraction.view(B, D, 1)
            + Z_addition.view(B, 1, S)
        )

        # forward rate
        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(B * D),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, D, S)

        # outer_qt0_numer_sig: B, x, S
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D * S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * D),
        ].view(B, D, S)

        # outer_qt0_denom_sig: B, x, tilde
        qt0_sig_norm_denom = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                minibatch.long().flatten(),
                x_tilde.long().flatten(),
            ].view(B, D)
            + self.ratio_eps
        )
        # sigma
        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask)
            / (Z_sig_norm * qt0_sig_norm_denom.view(B, D, 1)),
            dim=(1, 2),
        )

        sig_mean = torch.mean(-outer_sum_sig / sig_norm)

        reg_mean = torch.mean(reg_term)

        neg_elbo = sig_mean + reg_mean
        perm_x_logits = torch.permute(x_logits, (0, 2, 1))
        nll = self.cross_ent(perm_x_logits, minibatch.long())

        return neg_elbo + self.nll_weight * nll


@losses_utils.register_loss
class CondCTElbo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.condition_dim = cfg.loss.condition_dim
        self.cross_ent = nn.CrossEntropyLoss()

    def calc_loss(self, minibatch, state, writer=None):
        model = state["model"]
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)
        B, D = minibatch.shape
        device = model.device

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = model.transition(ts)  # (B, S, S)

        rate = model.rate(ts)  # (B, S, S)

        conditioner = minibatch[:, 0 : self.condition_dim]
        data = minibatch[:, self.condition_dim :]
        d = data.shape[1]

        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.flatten().long(),
            :,
        ]  # (B*d, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, d)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(d), x_t.long().flatten(), :
        ]  # (B*d, S)
        rate_vals_square[
            torch.arange(B * d, device=device), x_t.long().flatten()
        ] = 0.0  # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, d, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, d)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample()  # (B,) taking values in [0, d)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device), square_dims, :
        ]  # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = (
            square_newvalcat.sample()
        )  # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[torch.arange(B, device=device), square_dims] = square_newval_samples
        # x_tilde (B, d)

        # ---------- First term of ELBO (regularization) ---------------

        if self.one_forward_pass:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits_full = model(model_input, ts)  # (B, D, S)
            x_logits = x_logits_full[:, self.condition_dim :, :]  # (B, d, S)
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, d, S)
            reg_x = x_tilde
        else:
            model_input = torch.concat((conditioner, x_t), dim=1)
            x_logits_full = model(model_input, ts)  # (B, D, S)
            x_logits = x_logits_full[:, self.condition_dim :, :]  # (B, d, S)
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, d, S)
            reg_x = x_t

        mask_reg = torch.ones((B, d, S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            reg_x.long().flatten(),
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)

        qt0_denom_reg = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                :,
                reg_x.long().flatten(),
            ].view(B, d, S)
            + self.ratio_eps
        )

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            reg_x.long().flatten(),
        ].view(B, d, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(
            1, 2
        )  # (B, d, S)

        reg_term = torch.sum((p0t_reg / qt0_denom_reg) * reg_tmp, dim=(1, 2))

        # ----- second term of continuous ELBO (signal term) ------------

        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits_full = model(model_input, ts)  # (B, d, S)
            x_logits = x_logits_full[:, self.condition_dim :, :]
            p0t_sig = F.softmax(x_logits, dim=2)  # (B, d, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d * S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * d),
        ].view(B, d, S)

        outer_qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                data.long().flatten(),
                x_tilde.long().flatten(),
            ]
            + self.ratio_eps
        )  # (B, d)

        qt0_numer_sig = qt0.view(B, S, S)  # first S is x_0, second S is x

        qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                :,
                x_tilde.long().flatten(),
            ].view(B, d, S)
            + self.ratio_eps
        )

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        )  # (B, d, S)

        x_tilde_mask = torch.ones((B, d, S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            x_tilde.long().flatten(),
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(d * S),
            torch.arange(S, device=device).repeat(B * d),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, d, S)

        outer_sum_sig = torch.sum(
            x_tilde_mask
            * outer_rate_sig
            * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B, d, 1))
            * inner_log_sig,
            dim=(1, 2),
        )

        # now getting the 2nd term normalization

        rate_row_sums = -rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B),
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(d),
            x_tilde.long().flatten(),
        ].view(B, d)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp  # (B,d)
        Z_addition = rate_row_sums

        Z_sig_norm = (
            base_Z.view(B, 1, 1)
            - Z_subtraction.view(B, d, 1)
            + Z_addition.view(B, 1, S)
        )

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(d * S),
            torch.arange(S, device=device).repeat(B * d),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, d, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(d * S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * d),
        ].view(B, d, S)

        qt0_sig_norm_denom = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                data.long().flatten(),
                x_tilde.long().flatten(),
            ].view(B, d)
            + self.ratio_eps
        )

        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask)
            / (Z_sig_norm * qt0_sig_norm_denom.view(B, d, 1)),
            dim=(1, 2),
        )

        sig_mean = torch.mean(-outer_sum_sig / sig_norm)
        reg_mean = torch.mean(reg_term)

        neg_elbo = sig_mean + reg_mean

        perm_x_logits = torch.permute(x_logits, (0, 2, 1))

        nll = self.cross_ent(perm_x_logits, data.long())

        return neg_elbo + self.nll_weight * nll


# checked
@losses_utils.register_loss
class NLL:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.min_time = cfg.loss.min_time
        self.S = cfg.data.S
        self.D = cfg.model.concat_dim
        self.cross_ent = nn.CrossEntropyLoss()
        self.max_t = cfg.training.max_t

    def calc_loss(self, state, minibatch, label=None):
        """
        ce > 0 == ce < 0 + direct + rm

        Args:
            minibatch (_type_): _description_
            state (_type_): _description_
            writer (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        model = state["model"]

        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)

        B = minibatch.shape[0]
        device = self.cfg.device
        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time
        # ts = torch.clamp(ts, max=0.99999)

        qt0 = model.transition(ts)  # (B, S, S)

        b = utils.expand_dims(
            torch.arange(B, device=device), (tuple(range(1, minibatch.dim())))
        )
        qt0 = qt0[b, minibatch.long()].view(-1, self.S)  # B*D, S

        log_qt0 = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))
        xt = (
            torch.distributions.categorical.Categorical(logits=log_qt0)
            .sample()
            .view(B, self.D)
        )  # B, D

        # get logits from CondFactorizedBackwardModel
        logits = model(
            xt, ts, label
        )  # B, D, S: logits for every class in every dimension in x_t
        perm_x_logits = torch.permute(logits, (0, 2, 1))
        nll = self.cross_ent(perm_x_logits, minibatch.long())
        return nll  # sum over D

