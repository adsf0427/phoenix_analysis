import json
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *
from libriichi.consts import ACTION_SPACE


class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device=None,
        stochastic_latent=False,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name="NoName",
        boltzmann_epsilon=0,
        boltzmann_temp=1,
        top_p=1,
    ):
        self.engine_type = "mortal"
        self.device = device or torch.device("cpu")
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs):
        with (
            torch.autocast(self.device.type, enabled=self.enable_amp),
            torch.no_grad(),
        ):
            return self._react_batch(obs, masks, invisible_obs)

    def _react_batch(self, obs, masks, invisible_obs, only_q=False):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        invisible_obs = None
        if self.is_oracle:
            invisible_obs = torch.as_tensor(
                np.stack(invisible_obs, axis=0), device=self.device
            )
        batch_size = obs.shape[0]

        # match self.version:
        #     case 1:
        #         mu, logsig = self.brain(obs, invisible_obs)
        #         if self.stochastic_latent:
        #             latent = Normal(mu, logsig.exp() + 1e-6).sample()
        #         else:
        #             latent = mu
        #         q_out = self.dqn(latent, masks)
        #     case 2 | 3 | 4:
        assert self.version != 1
        phi = self.brain(obs)
        q_out = self.dqn(phi, masks)

        if only_q:
            return q_out

        if self.boltzmann_epsilon > 0:
            is_greedy = (
                torch.full(
                    (batch_size,), 1 - self.boltzmann_epsilon, device=self.device
                )
                .bernoulli()
                .to(torch.bool)
            )
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()


def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled


class MultiMortalEngine:
    def __init__(
        self,
        engines,
        weights,
        is_oracle,
        version,
        device=None,
        stochastic_latent=False,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name="NoName",
        boltzmann_epsilon=0,
        boltzmann_temp=1,
        top_p=1,
    ):
        self.engine_type = "mortal"
        self.device = device or torch.device("cpu")

        self.engines = engines
        self.weights = torch.tensor(weights, device=device or torch.device("cpu"))
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

        assert all(isinstance(engine, MortalEngine) for engine in engines)
        assert len(engines) == len(weights)

    def react_batch(self, obs, masks, invisible_obs):
        with (
            torch.autocast(self.device.type, enabled=self.enable_amp),
            torch.no_grad(),
        ):
            return self._react_batch(obs, masks, invisible_obs)

    def _react_batch(self, obs, masks, invisible_obs, only_q=False):
        batch_size = len(obs)
        reweighted_q_out = torch.zeros((batch_size, ACTION_SPACE), device=self.device)

        for engine, weight in zip(self.engines, self.weights):
            q_out = engine._react_batch(obs, masks, invisible_obs, only_q=True)
            reweighted_q_out += q_out * weight
        
        if only_q:
            return reweighted_q_out

        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if self.engines[0].boltzmann_epsilon > 0:
            is_greedy = (
                torch.full(
                    (batch_size,),
                    1 - self.engines[0].boltzmann_epsilon,
                    device=self.device,
                )
                .bernoulli()
                .to(torch.bool)
            )
            logits = (reweighted_q_out / self.engines[0].boltzmann_temp).masked_fill(
                ~masks, -torch.inf
            )
            sampled = sample_top_p(logits, self.engines[0].top_p)
            actions = torch.where(is_greedy, reweighted_q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = reweighted_q_out.argmax(-1)

        # print(f"mask type: {type(masks.tolist())}")
        return (
            actions.tolist(),
            reweighted_q_out.tolist(),
            masks.tolist(),
            is_greedy.tolist(),
        )

class SouthEastMortalEngine:
    def __init__(
        self,
        engine_south,
        engine_east,
        is_oracle,
        version,
        device=None,
        stochastic_latent=False,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name="NoName",
        boltzmann_epsilon=0,
        boltzmann_temp=1,
        top_p=1,
    ):
        self.engine_type = "mortal"
        self.device = device or torch.device("cpu")
        assert isinstance(self.device, torch.device)
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent
        
        self.engine_south = engine_south
        self.engine_east = engine_east

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs):
        with (
            torch.autocast(self.device.type, enabled=self.enable_amp),
            torch.no_grad(),
        ):
            return self._react_batch(obs, masks, invisible_obs)

    def _react_batch(self, obs, masks, invisible_obs, only_q=False):        
        south_q = self.engine_south._react_batch(obs, masks, invisible_obs, only_q=True)
        east_q = self.engine_east._react_batch(obs, masks, invisible_obs, only_q=True)
        
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        batch_size = obs.shape[0]

        is_east = obs[:, 25, 27] == 1.0        
        
        q_out = torch.where(is_east.unsqueeze(-1), east_q, south_q)

        if only_q:
            return q_out

        if self.boltzmann_epsilon > 0:
            is_greedy = (
                torch.full(
                    (batch_size,), 1 - self.boltzmann_epsilon, device=self.device
                )
                .bernoulli()
                .to(torch.bool)
            )
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

class CalibratedMortalEngine:
    def __init__(
        self,
        main_engine: MortalEngine,
        reference_engine: MortalEngine,
        known_engine: MortalEngine,
        calibration_prob,
        is_oracle,
        version,
        device=None,
        stochastic_latent=False,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name="NoName",
    ):
        self.engine_type = "mortal"
        self.device = device or torch.device("cpu")
        self.main_engine = main_engine
        self.reference_engine = reference_engine
        self.known_engine = known_engine
        self.calibration_prob = calibration_prob

        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

    def react_batch(self, obs, masks, invisible_obs):
        with (
            torch.autocast(self.device.type, enabled=self.enable_amp),
            torch.no_grad(),
        ):
            return self._react_batch(obs, masks, invisible_obs)

    def _react_batch(self, obs, masks, invisible_obs):        
        main_q = self.main_engine._react_batch(obs, masks, invisible_obs, only_q=True)
        ref_q = self.reference_engine._react_batch(
            obs, masks, invisible_obs, only_q=True
        )
        known_q = self.known_engine._react_batch(obs, masks, invisible_obs, only_q=True)
        
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        batch_size = len(obs)
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        main_actions = main_q.argmax(-1)
        ref_actions = ref_q.argmax(-1)
        known_actions = known_q.argmax(-1)

        calibration_mask = (
            (main_actions == ref_actions)
            & (main_actions != known_actions)
            & (torch.rand(batch_size, device=self.device) < self.calibration_prob)
            & (~masks[:, -1]) # -1 is the none(skip) action
            & (~masks[:, 37]) # 37 is the riichi action
        )
        calibration_mask = calibration_mask.unsqueeze(-1)
        q_out = torch.where(calibration_mask, known_q, main_q)
        actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()


class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = "mjai-log"
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]["type"] == "start_kyoku"

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": player_id,
                            "pai": tile,
                            "tsumogiri": True,
                        }
                    )
                )
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass

    def end_kyoku(self, game_idx: int):
        pass

    def end_game(self, game_idx: int, scores: List[int]):
        pass


if __name__ == "__main__":
    pass
    # N = 10
    # q_shape = (N, 46)
    # main_q = torch.randn(q_shape)
    # ref_q = torch.randn(q_shape)
    # known_q = torch.randn(q_shape)

    # batch_size = N

    # main_actions = main_q.argmax(-1)
    # ref_actions = ref_q.argmax(-1)
    # known_actions = known_q.argmax(-1)
    # print(main_actions)
    # print(ref_actions)
    # print(known_actions)

    # calibration_mask = (
    #     (main_actions == ref_actions)
    #     & (main_actions != known_actions)
    #     & (torch.rand(N) < 0.5)
    # )
    # print(calibration_mask)
    # calibration_mask = calibration_mask.unsqueeze(-1)
    # q_out = torch.where(calibration_mask, known_q, main_q)

    # actions = q_out.argmax(-1)
    # print(actions)
