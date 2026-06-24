import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class CrossingExtractor:
    """
    CrossingExtractor

    Purpose:
        Detect line crossings and estimate crossing times in a robust way
        (handles missed detections and ID switches via TTL logic).

    =================================================
    CONFIGURATION INPUTS (constructor)
    =================================================
    line_ids: List[str]
        Ordered list of line identifiers (must match GeometryEngine order)

    ttl_seconds: float
        Time threshold to trigger fallback crossing when object disappears
        after being in vicinity of a line

    =================================================
    CACHED / HIDDEN ATTRIBUTES (persistent state)
    =================================================
    _state: Dict[int, Dict[track_id, dict]]
        Per-line, per-object temporal state

        Structure:
        {
            line_idx: {
                track_id: {
                    "sign": int,
                    "dist": float,
                    "last_seen_time": float,
                    "was_in_vicinity": bool,
                    "polygons": List[str]
                }
            }
        }

    =================================================
    PER-FRAME INPUTS (update method)
    =================================================
    track_ids: np.ndarray (N,)
        IDs of tracked objects in current frame

    current_time: float
        Timestamp (seconds)

    line_cache: dict
        {
            "sign": (L, N) int (-1, 0, +1),
            "distance": (L, N) float (positive),
            "vicinity_mask": (L, N) bool
        }

    polygon_cache: Dict[str, np.ndarray]
        polygon_id → (N,) bool mask

    =================================================
    PER-FRAME OUTPUTS
    =================================================
    crossed_masks: np.ndarray (L, N) bool
        True if object i crossed line l in this frame

    new_crossings: Dict[str, List[float]]
        polygon_id → sorted list of crossing timestamps
    """

    # =================================================
    # CONSTRUCTOR
    # =================================================
    def __init__(self, line_ids: List[str], ttl_seconds: float):

        # -------- CONFIGURATION --------
        self.line_ids: List[str] = line_ids
        self.ttl: float = ttl_seconds

        # -------- CACHED STATE --------
        self._state: Dict[int, Dict[int, dict]] = {
            i: {} for i in range(len(line_ids))
        }

    # =================================================
    # MAIN UPDATE (PER FRAME)
    # =================================================
    def update(
        self,
        track_ids: np.ndarray,              # (N,)
        current_time: float,
        line_cache: Dict[str, np.ndarray],
        polygon_cache: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:

        # -------- UNPACK LINE FEATURES --------
        sign = line_cache["sign"]              # (L,N)
        dist = line_cache["distance"]          # (L,N)
        vicinity = line_cache["vicinity_mask"] # (L,N)

        L, N = sign.shape

        # -------- OUTPUT STRUCTURES --------
        crossed_masks = np.zeros((L, N), dtype=bool)
        new_crossings: Dict[str, List[float]] = defaultdict(list)

        # -------- VECTORIZE POLYGON MASKS --------
        polygon_ids = list(polygon_cache.keys())
        polygon_masks = np.vstack([
            polygon_cache[pid] for pid in polygon_ids
        ])  # (P,N)

        active_ids = set(track_ids)

        # -------- PROCESS EACH LINE --------
        for l in range(L):
            self._process_line(
                line_idx=l,
                track_ids=track_ids,
                current_time=current_time,
                s_curr=sign[l],        # (N,)
                d_curr=dist[l],        # (N,)
                v_curr=vicinity[l],    # (N,)
                polygon_ids=polygon_ids,
                polygon_masks=polygon_masks,
                crossed_masks=crossed_masks,
                new_crossings=new_crossings
            )

        # -------- TTL CLEANUP --------
        self._cleanup(active_ids, current_time, new_crossings)

        # -------- SORT OUTPUT --------
        for pid in new_crossings:
            new_crossings[pid].sort()

        return crossed_masks, new_crossings

    # =================================================
    # INTERNAL: PROCESS ONE LINE
    # =================================================
    def _process_line(
        self,
        line_idx: int,
        track_ids: np.ndarray,        # (N,)
        current_time: float,
        s_curr: np.ndarray,           # (N,)
        d_curr: np.ndarray,           # (N,)
        v_curr: np.ndarray,           # (N,)
        polygon_ids: List[str],
        polygon_masks: np.ndarray,    # (P,N)
        crossed_masks: np.ndarray,    # (L,N)
        new_crossings: Dict[str, List[float]]
    ):
        """
        Processes one line:
        - detects sign change crossings
        - updates state
        - assigns crossings to ALL polygons
        """

        line_state = self._state[line_idx]

        for i, tid in enumerate(track_ids):

            curr_sign = s_curr[i]
            curr_dist = d_curr[i]
            curr_vic = v_curr[i]

            # -------- MULTI-AREA ASSIGNMENT --------
            p_idxs = np.where(polygon_masks[:, i])[0]
            curr_polys = [polygon_ids[p] for p in p_idxs]

            # -------- INIT --------
            if tid not in line_state:
                line_state[tid] = {
                    "sign": curr_sign,
                    "dist": curr_dist,
                    "last_seen_time": current_time,
                    "was_in_vicinity": bool(curr_vic),
                    "polygons": curr_polys
                }
                continue

            st = line_state[tid]

            prev_sign = st["sign"]
            prev_dist = st["dist"]
            prev_time = st["last_seen_time"]

            # -------- TRUE CROSSING --------
            if (prev_sign * curr_sign) < 0:

                denom = prev_dist + curr_dist
                alpha = prev_dist / denom if denom > 1e-6 else 0.5

                t_cross = prev_time + alpha * (current_time - prev_time)

                for pid in st["polygons"]:
                    new_crossings[pid].append(t_cross)

                crossed_masks[line_idx, i] = True
                del line_state[tid]
                continue

            # -------- STATE UPDATE --------
            st["sign"] = curr_sign
            st["dist"] = curr_dist
            st["last_seen_time"] = current_time

            if curr_vic:
                st["was_in_vicinity"] = True

            if curr_polys:
                st["polygons"] = curr_polys

    # =================================================
    # INTERNAL: TTL CLEANUP
    # =================================================
    def _cleanup(
        self,
        active_ids: set,
        current_time: float,
        new_crossings: Dict[str, List[float]]
    ):
        """
        Handles missing detections:
        - if object disappeared after being in vicinity
        - and TTL exceeded → approximate crossing
        """

        for line_idx, line_state in self._state.items():

            to_delete = []

            for tid, st in line_state.items():

                if tid in active_ids:
                    continue

                if st["was_in_vicinity"] and (
                    current_time - st["last_seen_time"] > self.ttl
                ):
                    for pid in st["polygons"]:
                        new_crossings[pid].append(st["last_seen_time"])

                    to_delete.append(tid)

            for tid in to_delete:
                del line_state[tid]