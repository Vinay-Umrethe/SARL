import os
import shutil
import time
import math
import random
import numpy as np
import torch
import bluesky as bs
from bluesky import stack, traf, tools

from plugins.Multi_Agent.GAT_DDPG import GAT_DDPG, ReplayBuffer
from plugins.Multi_Agent.Normalizer_GAT import AircraftStateNormalizer


class Config:
    def __init__(self):
        self.mode = 'train'

        self.base_path = r"./"

        self.output_path = os.path.join(self.base_path, "output", "result", "SARL")
        self.txt_path = os.path.join(self.output_path, "result.txt")
        self.npy_path = os.path.join(self.output_path, "result.npy")

        self.scn_source = os.path.join(self.base_path, "scenario", "Route3.scn")
        self.route_file = "./routes/case_study_init_1.npy"

        os.makedirs(self.output_path, exist_ok=True)
        self.model_save_path = os.path.join(self.output_path, "MoE_Agent")
        self.scn_log_dir = os.path.join(self.output_path, f"{self.mode}_scn")
        os.makedirs(self.scn_log_dir, exist_ok=True)

        self.dt = 6.0
        self.num_intruders = 5

        self.max_speed_delta = 4.5
        self.max_alt_delta = 300.0
        self.max_hdg_delta = 5.0

        self.limits = {
            'speed': (200, 300),
            'alt': (20000, 21000)
        }
        self.flight_levels = [alt for alt in range(20000, 21000, 300)]

        # --- RL ---
        self.shield_dropout_rate = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = 50000
        self.batch_size = 1024
        self.gamma = 0.99
        self.curriculum = [(float('inf'), 18, 2000, 50)]

        self.reward_weights = {
            'collision': -1000.0,
            'shield_intervention': -5.0,

            'track_error': -1.0,
            'track_sq_penalty': -0.1,
            'heading_error': -1.0,
            'progress': 1.0,
            'pos_progress': 0.5,

            'alt_hold': -1.0,
            'action_smooth': -0.5,

            'noop_bonus': 0.2,
            'expert_switch': -0.5,

            'boundary': -50.0,
            'goal_arrival': 200.0
        }

        self.reward_params = {
            'accept_xtk': 2.0,
            'accept_alt_err': 50.0,
            'arrival_dist': 5.0,
            'max_track_width': 20.0,
            'safe_dist': 10.0,

            'collision_h_km': 2.0,
            'collision_v_m': 300.0,

            'lookahead_time': 120.0,
            'min_sep_h': 9260.0,
            'min_sep_v': 1000.0
        }

class ACREnvironmentController:
    def __init__(self):
        self.cfg = Config()
        self.normalizer = AircraftStateNormalizer(num_intruders=self.cfg.num_intruders)
        self.replay_buffer = ReplayBuffer(self.cfg.buffer_size)

        self.agent = GAT_DDPG(
            ego_dim=20,
            neighbor_dim=15,
            hidden_dim=128,
            action_dim=3,
            device=self.cfg.device
        )

        self.active_experience = {}
        self.route_assignments = {}
        self.assigned_levels = {}
        self.routes = np.load(self.cfg.route_file)
        self.spawn_choices = [60, 70, 80]
        self.last_expert_selection = {}

        self.episode_count = 1
        self.step_count = 0
        self.num_ac_generated_total = 0
        self.win_count = 0
        self.flight_stats = {}
        self.efficiency_history = []
        self.effort_history = []
        self.safety_stats = {'los_count': 0, 'nmac_count': 0, 'min_sep_dist': 50.0}
        self.route_timers = [random.choice(self.spawn_choices) for _ in range(len(self.routes))]
        self.curr_max_conc, self.curr_max_steps, self.curr_total = self.cfg.curriculum[0][1:]

        self.global_stats = {
            'total_aircraft': [],
            'success_count': [],
            'collision_count': [],
            'boundary_count': [],

            'total_extra_dist_pct': [],
            'success_sample_count': [],

            'total_min_sep': [],
            'total_warn_frames': [],

            'total_cost': [],
            'cost_sample_count': []
        }
        self.episode_outcomes = []

        self.log_file_path = None
        self._init_log_file()


        print(f"\n=== Episode {self.episode_count} Started (Phase 2: CD&R Shield) ===")

    # def _load_pretrained_models(self):
    #     if os.path.exists(self.cfg.pretrained_model_path + "_actor.pth"):
    #         print(f"[Init] Loading pretrained model from Phase 1: {self.cfg.pretrained_model_path}")
    #         self.agent.load(self.cfg.pretrained_model_path)
    #     else:
    #         print("[Init] WARNING: No model found. Starting from RANDOM initialization.")

    def reset(self):
        self.episode_count += 1
        if self.episode_count == 5:
            stack.stack("QUIT")
        self.step_count = 0
        self.num_ac_generated_total = 0
        self.win_count = 0
        self.active_experience = {}
        self.route_assignments = {}
        self.assigned_levels = {}
        self.flight_stats = {}
        self.last_expert_selection = {}
        self.efficiency_history = []
        self.effort_history = []
        self.safety_stats = {'los_count': 0, 'nmac_count': 0, 'min_sep_dist': 50.0}

        self.episode_outcomes = []

        self.route_timers = [random.choice(self.spawn_choices) for _ in range(len(self.routes))]

        if self.cfg.mode == 'train' and self.replay_buffer.size() > self.cfg.batch_size:
            self._update_agent(10)

        print(f"\n=== Episode {self.episode_count} ===")
        self._init_log_file()
        stack.stack('IC DQN_3D.scn')
        if self.cfg.mode == 'train':
            self.agent.save(self.cfg.model_save_path)

    def step(self):
        self.step_count += 1
        all_spawned = (self.num_ac_generated_total >= self.curr_total)
        all_cleared = (len(traf.id) == 0)

        if self.step_count >= self.curr_max_steps or (all_spawned and all_cleared and self.step_count > 10):
            self._conclude_episode()
            self.reset()
            return

        self._spawn_traffic()
        if len(traf.id) == 0: return

        neighbor_map = self._get_neighbors_info_with_id()

        gat_inputs = self._collect_gat_states(neighbor_map)
        self._update_safety_stats(gat_inputs)

        current_raw_actions = {}
        current_step_experts = {}
        rewards = {}
        dones = {}
        infos = {}

        current_noise = 0.05 + (0.2 - 0.05) * max(0, 1-(self.episode_count/800))

        for acid, inputs in gat_inputs.items():
            noise = 0.2 if self.cfg.mode == 'train' else 0.0

            # --- Agent Decision ---
            raw_action, expert_idx = self.agent.take_action(inputs['ego'], inputs['neigh'], inputs['mask'],
                                                            noise_sigma=noise)
            current_step_experts[acid] = expert_idx

            # --- CD&R Shield ---
            neighbors_raw = neighbor_map.get(acid, [])
            safe_action, intervention_level = self._apply_phase2_shield(acid, raw_action, neighbors_raw)
            shield_triggered = (intervention_level > 0)

            # --- Execution ---
            self._apply_incremental_action(acid, safe_action)
            current_raw_actions[acid] = raw_action
            if acid in self.flight_stats: self.flight_stats[acid]['fuel_proxy'] += np.linalg.norm(raw_action)

            # --- Reward ---
            r, d, i = self._compute_reward_phase2(acid, raw_action, inputs, shield_triggered, expert_idx)
            rewards[acid] = r
            dones[acid] = d
            infos[acid] = i

        self._update_flight_metrics()

        if self.cfg.mode == 'train':
            for acid in self.active_experience:
                if acid in gat_inputs and acid in rewards:
                    last = self.active_experience[acid]
                    self.replay_buffer.add(
                        last['state']['ego'], last['state']['neigh'], last['state']['mask'],
                        last['action'], rewards[acid],
                        gat_inputs[acid]['ego'], gat_inputs[acid]['neigh'], gat_inputs[acid]['mask'],
                        dones.get(acid, False)
                    )

        self.active_experience = {}
        for acid in gat_inputs:
            if dones.get(acid, False):
                self._remove_aircraft(acid, infos.get(acid, ""))
                if acid in self.last_expert_selection: del self.last_expert_selection[acid]
                if acid in self.assigned_levels: del self.assigned_levels[acid]
            else:
                self.active_experience[acid] = {'state': gat_inputs[acid], 'action': current_raw_actions[acid]}
                if acid in current_step_experts: self.last_expert_selection[acid] = current_step_experts[acid]

        if self.step_count % 100 == 0: self._print_status()

    def _conclude_episode(self):
        active_acids = list(self.flight_stats.keys())
        for acid in active_acids:
            self._remove_aircraft(acid, "BOUNDARY")

        ep_total = len(self.episode_outcomes)
        if ep_total == 0: return

        self.global_stats['total_aircraft'].append(ep_total)

        warn_frames = 0.0
        min_sep = 0.0
        success_count = 0
        success_sample_count = 0
        extra_dist_pct = 0.0
        cost = 0.0
        cost_sample_count = 0
        collision_count = 0
        boundary_count = 0
        for outcome in self.episode_outcomes:
            res = outcome['result']

            warn_frames += outcome['warn_frames']
            min_sep += outcome['min_sep']

            if res == 'GOAL':
                success_count += 1
                success_sample_count += 1
                extra_dist_pct += outcome['extra_dist_pct']
                cost += outcome['cost']
                cost_sample_count += 1

            elif res == 'COLLISION':
                collision_count += 1
            else:
                boundary_count += 1

        self.global_stats['total_warn_frames'].append(warn_frames)
        self.global_stats['total_min_sep'].append(min_sep)

        self.global_stats['success_count'].append(success_count)
        self.global_stats['success_sample_count'].append(success_sample_count)
        self.global_stats['total_extra_dist_pct'].append(extra_dist_pct)
        self.global_stats['total_cost'].append(cost)
        self.global_stats['cost_sample_count'].append(cost_sample_count)

        self.global_stats['collision_count'].append(collision_count)
        self.global_stats['boundary_count'].append(boundary_count)

        total_ac = np.sum(self.global_stats['total_aircraft'][-100:])
        success_ac = np.sum(self.global_stats['success_sample_count'][-100:])

        avg_success = (np.sum(self.global_stats['success_count'][-100:]) / total_ac) * 100
        avg_col = (np.sum(self.global_stats['collision_count'][-100:]) / total_ac) * 100
        avg_bound = (np.sum(self.global_stats['boundary_count'][-100:]) / total_ac) * 100

        avg_extra_dist = 0.0
        if success_ac > 0:
            avg_extra_dist = (np.sum(self.global_stats['total_extra_dist_pct'][-100:]) / success_ac) * 100

        avg_min_sep = np.sum(self.global_stats['total_min_sep'][-100:]) / total_ac

        avg_warn = np.sum(self.global_stats['total_warn_frames'][-100:]) / total_ac

        avg_cost = 0.0
        if self.global_stats['cost_sample_count'][-1] > 0:
            avg_cost = np.sum(self.global_stats['total_cost'][-100:]) / np.sum(self.global_stats['cost_sample_count'][-100:])

        stats_output = f"""
        {"=" * 65}
        📊 CUMULATIVE STATS (Episodes {max(0, self.episode_count - 99)}-{self.episode_count})
           Total Aircraft Gen : {total_ac}
           --------------------------------------------------
           [Performance]
           1. Success Rate    : {avg_success:.2f}%
           2. Collision Rate  : {avg_col:.2f}%
           3. Boundary Rate   : {avg_bound:.2f}%
           --------------------------------------------------
           [Efficiency & Cost]
           4. Extra Dist %    : {avg_extra_dist:.2f}% (Success Flights Only)
           5. Avg Op. Cost    : {avg_cost:.4f} (Action Norm Sum)
           --------------------------------------------------
           [Safety Margins]
           6. Avg Min Sep     : {avg_min_sep:.3f} km (Global Safety Margin)
           7. Avg Warn Duration: {avg_warn:.2f} frames (< {self.cfg.reward_params['safe_dist']}km)
        {"=" * 65}
        """
        print(stats_output)
        self.save_global_stats_to_npy()

        with open(self.cfg.txt_path, "a", encoding="utf-8") as f:
            f.write(stats_output)
            f.write("\n")

    def _apply_phase2_shield(self, acid, raw_action, neighbors):
        safe_action = raw_action.copy()
        intervention_level = 0

        idx = traf.id2idx(acid)
        own_lat, own_lon = traf.lat[idx], traf.lon[idx]
        own_spd = traf.cas[idx] * 1.9439 * 0.514444
        own_trk = math.radians(traf.hdg[idx])
        own_alt_ft = traf.alt[idx] * 3.28084

        v_own_n = own_spd * math.cos(own_trk)
        v_own_e = own_spd * math.sin(own_trk)

        min_dcpa = 99999.0
        min_tcpa = 99999.0
        critical_neigh = None

        for neigh in neighbors:
            n_alt_ft = neigh[3]
            if abs(own_alt_ft - n_alt_ft) >= 300: continue

            n_lat, n_lon = neigh[0], neigh[1]
            n_spd_ms = neigh[2] * 0.514444
            n_trk_rad = math.radians(neigh[4])

            dist_km = MathUtils.calculate_haversine_distance(own_lat, own_lon, n_lat, n_lon)
            bearing_rad = math.radians(MathUtils.calculate_bearing(own_lat, own_lon, n_lat, n_lon))
            p_rel_n = dist_km * 1000.0 * math.cos(bearing_rad)
            p_rel_e = dist_km * 1000.0 * math.sin(bearing_rad)

            v_neigh_n = n_spd_ms * math.cos(n_trk_rad)
            v_neigh_e = n_spd_ms * math.sin(n_trk_rad)
            v_rel_n = v_neigh_n - v_own_n
            v_rel_e = v_neigh_e - v_own_e

            t_cpa, d_cpa = MathUtils.calculate_cpa(p_rel_n, p_rel_e, v_rel_n, v_rel_e)

            if 0 < t_cpa < self.cfg.reward_params['lookahead_time']:
                if d_cpa < min_dcpa:
                    min_dcpa = d_cpa
                    min_tcpa = t_cpa
                    critical_neigh = neigh

        SAFE_SEP = 5000.0
        if min_dcpa < SAFE_SEP:
            if min_dcpa < 1000.0 or min_tcpa < 30.0:
                intervention_level = 2
                own_id_val = int(acid.replace('KL', ''))
                try:
                    other_id_val = int(critical_neigh[5].replace('KL', ''))
                except:
                    other_id_val = 0

                if own_id_val > other_id_val:
                    safe_action[1] = 1.0
                else:
                    safe_action[1] = -1.0
                safe_action[0] = 0.0
                safe_action[2] = 0.0
            else:
                intervention_level = 1
                safe_action[2] = 1.0
                safe_action[0] = -0.2

        pred_alt = own_alt_ft + (safe_action[1] * self.cfg.max_alt_delta)
        if pred_alt < 20000 and safe_action[1] < -0.1:
            safe_action[1] = 0.0
        elif pred_alt > 21000 and safe_action[1] > 0.1:
            safe_action[1] = 0.0

        return safe_action, intervention_level

    def _compute_reward_phase2(self, acid, action, inputs, shield_triggered, current_expert):
        w = self.cfg.reward_weights
        p = self.cfg.reward_params
        idx = traf.id2idx(acid)
        route_idx = self.route_assignments.get(acid, 0)
        route = self.routes[route_idx]

        last_lat, last_lon = self.flight_stats[acid]['prev_pos']
        lat, lon = traf.lat[idx], traf.lon[idx]
        target_alt = self.assigned_levels.get(acid, 20000.0)
        curr_alt_ft = traf.alt[idx] * 3.28084

        r, done, info = 0.0, False, ""

        if shield_triggered:
            r += w['shield_intervention']

        if inputs['collision_flag'] > 0.5:
            r += w['collision']
            info = "COLLISION"
            done = True

        xtk = MathUtils.calculate_distance_to_line(lat, lon, route[0], route[1], route[2], route[3])
        r += w['track_error'] * xtk
        r += w['track_sq_penalty'] * (xtk ** 2)

        route_heading = route[4]
        dist_from_start = MathUtils.calculate_haversine_distance(route[0], route[1], lat, lon)
        bearing_from_start = MathUtils.calculate_bearing(route[0], route[1], lat, lon)
        angle_diff = math.radians(bearing_from_start - route_heading)
        xtk_signed = dist_from_start * math.sin(angle_diff)

        intercept_angle = math.degrees(math.atan(-5.0 * xtk_signed))
        desired_heading = (route_heading + intercept_angle) % 360
        curr_hdg = traf.hdg[idx]
        vf_hdg_err = abs((desired_heading - curr_hdg + 180) % 360 - 180)
        r += w['heading_error'] * (vf_hdg_err / 180.0)

        along_track_dist = dist_from_start * math.cos(angle_diff)
        last_dist_from_start = MathUtils.calculate_haversine_distance(route[0], route[1], last_lat, last_lon)
        last_bearing_from_start = MathUtils.calculate_bearing(route[0], route[1], last_lat, last_lon)
        last_angle_diff = math.radians(last_bearing_from_start - route_heading)
        last_along_track_dist = last_dist_from_start * math.cos(last_angle_diff)
        dist_improv = along_track_dist - last_along_track_dist
        if dist_improv > 0:
            r += min(dist_improv * w['progress'], 5.0)
        else:
            r -= abs(dist_improv) * w['pos_progress']

        alt_diff = abs(curr_alt_ft - target_alt)
        r += w['alt_hold'] * (alt_diff / 100.0)
        r += w['action_smooth'] * np.mean(np.abs(action))

        dist_to_goal = MathUtils.calculate_haversine_distance(lat, lon, route[2], route[3])
        if xtk > p['max_track_width']:
            r += w['boundary']
            done = True
            info = "BOUNDARY"
        elif dist_to_goal < p['arrival_dist']:
            r += w['goal_arrival']
            done = True
            info = "GOAL"
            self.win_count += 1

        r = max(-500.0, min(r, 200.0))
        return r, done, info

    def _collect_gat_states(self, neighbor_map):
        states = {}
        raw_states = self._collect_raw_states()

        for acid in raw_states:
            route_idx = self.route_assignments.get(acid, 0)
            route = self.routes[route_idx]
            route_info = (route[0], route[1], route[2], route[3])
            target_alt = self.assigned_levels.get(acid, 20000.0)

            idx = traf.id2idx(acid)
            own_lat, own_lon = traf.lat[idx], traf.lon[idx]
            own_alt_ft = traf.alt[idx] * 3.28084

            is_in_cylinder = 0.0
            min_dist_val = 50.0

            raw_neighs = neighbor_map.get(acid, [])

            for i, neigh in enumerate(raw_neighs):
                n_lat, n_lon = neigh[0], neigh[1]
                n_alt_ft = neigh[3]

                h_dist_km = MathUtils.calculate_haversine_distance(own_lat, own_lon, n_lat, n_lon)
                v_dist_ft = abs(own_alt_ft - n_alt_ft)

                if h_dist_km < min_dist_val and v_dist_ft < self.cfg.reward_params['collision_v_m']: min_dist_val = h_dist_km

                if h_dist_km < self.cfg.reward_params['collision_h_km'] and v_dist_ft < 300.0:
                    is_in_cylinder = 1.0

                if v_dist_ft < 300.0:
                    raw_neighs[i].append(1.0)
                else:
                    raw_neighs[i].append(0.0)

            norm_ego, route_heading_ref = self.normalizer.normalize_own_state_with_route(
                raw_states[acid], route_info, target_alt, 0  # ego flag unused
            )
            norm_neighs_flat = self.normalizer.normalize_other_aircraft(
                acid, raw_neighs, raw_states[acid], route_heading_ref
            )

            neigh_matrix = np.array(norm_neighs_flat).reshape(self.cfg.num_intruders, 15)
            mask = np.zeros(self.cfg.num_intruders)
            mask[:len(raw_neighs)] = 1.0

            states[acid] = {
                'ego': np.array(norm_ego),
                'neigh': neigh_matrix,
                'mask': mask,
                'min_dist_km': min_dist_val,
                'collision_flag': is_in_cylinder
            }
        return states

    def _get_neighbors_info_with_id(self):
        if len(traf.id) == 0: return {}
        neighbor_map = {}
        lats, lons, spds, alts, hdgs = traf.lat, traf.lon, traf.cas, traf.alt, traf.hdg
        for i, acid_i in enumerate(traf.id):
            dist_list = []
            for j, acid_j in enumerate(traf.id):
                if i == j: continue
                d = MathUtils.calculate_haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if d < 50.0: dist_list.append((d, j))
            dist_list.sort(key=lambda x: x[0])
            top_n = dist_list[:self.cfg.num_intruders]
            n_data = []
            for _, k in top_n: n_data.append(
                [lats[k], lons[k], spds[k] * 1.9439, alts[k] * 3.28084, hdgs[k], traf.id[k]])
            neighbor_map[acid_i] = n_data
        return neighbor_map

    def _collect_raw_states(self):
        states = {}
        for i, acid in enumerate(traf.id):
            idx = traf.id2idx(acid)
            route_idx = self.route_assignments.get(acid, 0)
            route = self.routes[route_idx]
            lat_now, lon_now = traf.lat[idx], traf.lon[idx]
            xtk = MathUtils.calculate_distance_to_line(lat_now, lon_now, route[0], route[1], route[2], route[3])
            dist = MathUtils.calculate_haversine_distance(lat_now, lon_now, route[2], route[3])
            b_goal = MathUtils.calculate_bearing(lat_now, lon_now, route[2], route[3])
            hdg_err = (b_goal - traf.hdg[idx] + 180) % 360 - 180
            s = [lat_now, lon_now, traf.cas[idx] * 1.9439, traf.alt[idx] * 3.28084, traf.hdg[idx], xtk, dist, b_goal,
                 hdg_err]
            states[acid] = s
        return states

    def _apply_incremental_action(self, acid, action):
        idx = traf.id2idx(acid)
        d_spd = action[0] * self.cfg.max_speed_delta
        d_alt = action[1] * self.cfg.max_alt_delta
        d_hdg = action[2] * self.cfg.max_hdg_delta
        curr_spd = traf.cas[idx] * 1.9439
        curr_alt = traf.alt[idx] * 3.28084
        curr_hdg = traf.hdg[idx]
        target_spd = MathUtils.clamp(curr_spd + d_spd, *self.cfg.limits['speed'])
        target_alt = MathUtils.clamp(curr_alt + d_alt, *self.cfg.limits['alt'])
        target_hdg = (curr_hdg + d_hdg) % 360
        stack.stack(f'SPD {acid} {target_spd}')
        stack.stack(f'ALT {acid} {target_alt}')
        stack.stack(f'HDG {acid} {target_hdg}')
        self._write_scn_log(f'SPD {acid} {target_spd:.1f}')
        self._write_scn_log(f'ALT {acid} {target_alt:.0f}')
        self._write_scn_log(f'HDG {acid} {target_hdg:.1f}')

    def _spawn_traffic(self):
        if self.num_ac_generated_total >= self.curr_total: return
        if len(traf.id) == 0:
            for i in range(len(self.routes)):
                if len(traf.id) >= self.curr_max_conc: break
                if self.num_ac_generated_total >= self.curr_total: break
                self._create_aircraft(i)
                self.route_timers[i] = self.step_count + random.choice(self.spawn_choices)
        else:
            gen = 0
            for k in range(len(self.route_timers)):
                if self.step_count == self.route_timers[k]:
                    if len(traf.id) + gen >= self.curr_max_conc:
                        self.route_timers = [t + 10 for t in self.route_timers]
                        continue
                    gen += 1
                    self._create_aircraft(k)
                    self.route_timers[k] = self.step_count + random.choice(self.spawn_choices)
                    if self.num_ac_generated_total >= self.curr_total: break

    def _create_aircraft(self, route_idx):
        lat, lon, glat, glon, _ = self.routes[route_idx]
        acid = f"KL{self.num_ac_generated_total}"
        init_alt = random.choice(self.cfg.flight_levels)
        init_hdg = MathUtils.calculate_bearing(lat, lon, glat, glon)
        stack.stack(f'CRE {acid}, B737, {lat}, {lon}, {init_hdg}, {init_alt}, 250')
        stack.stack(f'ADDWPT {acid} {glat}, {glon}')
        stack.stack(f'VNAV {acid} ON')
        self._write_scn_log(f'CRE {acid}, B737, {lat}, {lon}, {init_hdg}, {init_alt}, 250')
        self._write_scn_log(f'ADDWPT {acid} {glat}, {glon}')
        self._write_scn_log(f'VNAV {acid} ON')
        self.route_assignments[acid] = route_idx
        self.assigned_levels[acid] = float(init_alt)
        self.num_ac_generated_total += 1
        ideal_dist = MathUtils.calculate_haversine_distance(lat, lon, glat, glon)
        self.flight_stats[acid] = {
            'dist_flown': 0.0, 'ideal_dist': ideal_dist, 'fuel_proxy': 0.0, 'prev_pos': (lat, lon),
            'min_sep': 50.0, 'warn_frames': 0, "ori": (lat, lon), "goal": (glat, glon), "counts": 0
        }

    def _update_flight_metrics(self):
        for i, acid in enumerate(traf.id):
            if acid in self.flight_stats:
                curr_lat, curr_lon = traf.lat[i], traf.lon[i]
                prev_lat, prev_lon = self.flight_stats[acid]['prev_pos']
                self.flight_stats[acid]['dist_flown'] += MathUtils.calculate_haversine_distance(prev_lat, prev_lon,
                                                                                                curr_lat, curr_lon)
                self.flight_stats[acid]['prev_pos'] = (curr_lat, curr_lon)
                self.flight_stats[acid]['counts'] += 1

    def _update_safety_stats(self, gat_inputs):
        min_d = 50.0
        for acid, data in gat_inputs.items():
            d = data.get('min_dist_km', 50.0)
            if d < min_d: min_d = d
            if d < self.cfg.reward_params['safe_dist']: self.safety_stats['los_count'] += 1
            if d < self.cfg.reward_params['collision_h_km']: self.safety_stats['nmac_count'] += 1
            if acid in self.flight_stats:
                self.flight_stats[acid]['min_sep'] = min(self.flight_stats[acid]['min_sep'], d)
                if d < self.cfg.reward_params['safe_dist']:
                    self.flight_stats[acid]['warn_frames'] += 1
        if min_d < self.safety_stats['min_sep_dist']: self.safety_stats['min_sep_dist'] = min_d

    def _remove_aircraft(self, acid, reason):
        stack.stack(f'DEL {acid}')
        self._write_scn_log(f'DEL {acid}')
        if acid in self.flight_stats:
            idx = traf.id2idx(acid)
            stats = self.flight_stats[acid]
            te = stats['ideal_dist'] / max(stats['dist_flown'], 0.1)
            extra_dist_pct = 0.0
            if reason == 'GOAL':
                _lat, _lon = stats['ori']
                _glat, _glon = stats['goal']
                lat = np.linspace(_lat, _glat, stats['counts'])
                lon = np.linspace(_lon, _glon, stats['counts'])
                ideal_dist = 0
                for i in range(stats['counts'] - 1):
                    ideal_dist += MathUtils.calculate_haversine_distance(lat[i], lon[i], lat[i + 1], lon[i + 1])
                diff = stats['dist_flown'] - ideal_dist + MathUtils.calculate_haversine_distance(traf.lat[idx],
                                                                                                 traf.lon[idx], _glat,
                                                                                                 _glon)
                extra_dist_pct = (max(0.0, diff) / stats['ideal_dist']) * 100

            self.efficiency_history.append(te)
            self.effort_history.append(stats['fuel_proxy'])

            self.episode_outcomes.append({
                'acid': acid,
                'result': reason,
                'cost': stats['fuel_proxy'],
                'warn_frames': stats['warn_frames'],
                'min_sep': stats['min_sep'],
                'extra_dist_pct': extra_dist_pct
            })

            if reason == "GOAL":
                print(f"✈️ {acid} Arrived. TE={te:.3f}")
            elif reason == "COLLISION":
                print(f"💥 {acid} Collided!")
            elif reason == "BOUNDARY":
                print(f"❌ {acid} Out of Bound!")
            del self.flight_stats[acid]

    def _print_episode_stats(self):
        avg_te = np.mean(self.efficiency_history) if self.efficiency_history else 0.0
        avg_effort = np.mean(self.effort_history) if self.effort_history else 0.0
        print(
            f"\n📊 Episode {self.episode_count} Summary: Gen {self.num_ac_generated_total}/{self.curr_total} | Success {self.win_count} | TE {avg_te:.4f} | MinSep {self.safety_stats['min_sep_dist']:.3f}")

    def _print_status(self):
        print(
            f"Step {self.step_count}: AC {len(traf.id)} | Gen {self.num_ac_generated_total} | MinSep {self.safety_stats['min_sep_dist']:.2f}")

    def _init_log_file(self):
        if self.cfg.mode == 'eval' or (self.cfg.mode == 'train' and self.episode_count % 100 == 0):
            self.log_file_path = os.path.join(self.cfg.scn_log_dir, f"Ep{self.episode_count}.scn")
            shutil.copy2(self.cfg.scn_source, self.log_file_path)
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n# Episode {self.episode_count} Log\n")
        else:
            self.log_file_path = None

    def _write_scn_log(self, text):
        if self.log_file_path:
            t_str = time.strftime('%H:%M:%S.00', time.gmtime(bs.sim.simt))
            with open(self.log_file_path, 'a', encoding='utf-8') as f: f.write(f"{t_str}>{text}\n")

    def _update_agent(self, round=2):
        for _ in range(round):
            batch = self.replay_buffer.sample(self.cfg.batch_size)
            self.agent.update(
                {'ego': batch[0], 'neigh': batch[1], 'mask': batch[2], 'action': batch[3], 'reward': batch[4],
                 'next_ego': batch[5], 'next_neigh': batch[6], 'next_mask': batch[7], 'done': batch[8]})

    def save_global_stats_to_npy(self):
        dtype = [
            ('episode', 'int32'),
            ('total_aircraft', 'int32'),
            ('success_count', 'int32'),
            ('collision_count', 'int32'),
            ('boundary_count', 'int32'),
            ('total_extra_dist_pct', 'float32'),
            ('success_sample_count', 'int32'),
            ('total_min_sep', 'float32'),
            ('total_warn_frames', 'int32'),
            ('total_cost', 'float32'),
            ('cost_sample_count', 'int32')
        ]

        n_episodes = len(self.global_stats['total_aircraft'])

        structured_array = np.zeros(n_episodes, dtype=dtype)

        structured_array['episode'] = np.arange(n_episodes)
        structured_array['total_aircraft'] = np.array(self.global_stats['total_aircraft'], dtype='int32')
        structured_array['success_count'] = np.array(self.global_stats['success_count'], dtype='int32')
        structured_array['collision_count'] = np.array(self.global_stats['collision_count'], dtype='int32')
        structured_array['boundary_count'] = np.array(self.global_stats['boundary_count'], dtype='int32')
        structured_array['total_extra_dist_pct'] = np.array(self.global_stats['total_extra_dist_pct'], dtype='float32')
        structured_array['success_sample_count'] = np.array(self.global_stats['success_sample_count'], dtype='int32')
        structured_array['total_min_sep'] = np.array(self.global_stats['total_min_sep'], dtype='float32')
        structured_array['total_warn_frames'] = np.array(self.global_stats['total_warn_frames'], dtype='int32')
        structured_array['total_cost'] = np.array(self.global_stats['total_cost'], dtype='float32')
        structured_array['cost_sample_count'] = np.array(self.global_stats['cost_sample_count'], dtype='int32')

        np.save(self.cfg.npy_path, structured_array)

        return structured_array


class MathUtils:
    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
            dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    @staticmethod
    def calculate_distance_to_line(plat, plon, lat1, lon1, lat2, lon2):
        d13 = MathUtils.calculate_haversine_distance(lat1, lon1, plat, plon) / 6371.0
        brng13 = math.radians(MathUtils.calculate_bearing(lat1, lon1, plat, plon))
        brng12 = math.radians(MathUtils.calculate_bearing(lat1, lon1, lat2, lon2))
        return abs(math.asin(math.sin(d13) * math.sin(brng13 - brng12))) * 6371.0

    @staticmethod
    def clamp(val, min_v, max_v):
        return max(min_v, min(val, max_v))

    @staticmethod
    def calculate_cpa(p_rel_n, p_rel_e, v_rel_n, v_rel_e):
        v_rel_sq = v_rel_n ** 2 + v_rel_e ** 2
        if v_rel_sq < 1e-6: return 0.0, math.sqrt(p_rel_n ** 2 + p_rel_e ** 2)
        t_cpa = -(p_rel_n * v_rel_n + p_rel_e * v_rel_e) / v_rel_sq
        if t_cpa <= 0:
            d_cpa = math.sqrt(p_rel_n ** 2 + p_rel_e ** 2)
        else:
            p_cpa_n = p_rel_n + v_rel_n * t_cpa
            p_cpa_e = p_rel_e + v_rel_e * t_cpa
            d_cpa = math.sqrt(p_cpa_n ** 2 + p_cpa_e ** 2)
        return t_cpa, d_cpa


# Plugin Entry
controller = None


def init_plugin():
    global controller
    config = {'plugin_name': 'case_step3_SARL', 'plugin_type': 'sim', 'update_interval': 6.0,
              'update': update}
    controller = ACREnvironmentController()
    return config, {}


def update():
    if controller: controller.step()