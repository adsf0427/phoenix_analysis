use super::{ActionCandidate, PlayerState};
use crate::tile::Tile;

use pyo3::prelude::*;

#[pymethods]
impl PlayerState {
    #[getter]
    #[inline]
    #[must_use]
    pub const fn player_id(&self) -> u8 {
        self.player_id
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyoku(&self) -> u8 {
        self.kyoku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn honba(&self) -> u8 {
        self.honba
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyotaku(&self) -> u8 {
        self.kyotaku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn is_oya(&self) -> bool {
        self.oya == 0
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn tiles_left(&self) -> u8 {
        self.tiles_left
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tehai(&self) -> [u8; 34] {
        self.tehai
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }

    #[getter]
    #[inline]
    #[must_use]
    pub fn chis(&self) -> &[u8] {
        &self.chis
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn pons(&self) -> &[u8] {
        &self.pons
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn minkans(&self) -> &[u8] {
        &self.minkans
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn ankans(&self) -> &[u8] {
        &self.ankans
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_turn(&self) -> u8 {
        self.at_turn
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn shanten(&self) -> i8 {
        self.shanten
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn waits(&self) -> [bool; 34] {
        self.waits
    }

    #[inline]
    #[pyo3(name = "last_self_tsumo")]
    fn last_self_tsumo_py(&self) -> Option<String> {
        self.last_self_tsumo.map(|t| t.to_string())
    }
    #[inline]
    #[pyo3(name = "last_kawa_tile")]
    fn last_kawa_tile_py(&self) -> Option<String> {
        self.last_kawa_tile.map(|t| t.to_string())
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn last_cans(&self) -> ActionCandidate {
        self.last_cans
    }

    #[inline]
    #[pyo3(name = "ankan_candidates")]
    fn ankan_candidates_py(&self) -> Vec<String> {
        self.ankan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }
    #[inline]
    #[pyo3(name = "kakan_candidates")]
    fn kakan_candidates_py(&self) -> Vec<String> {
        self.kakan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_w_riichi(&self) -> bool {
        self.can_w_riichi
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn self_riichi_declared(&self) -> bool {
        self.riichi_declared[0]
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn self_riichi_accepted(&self) -> bool {
        self.riichi_accepted[0]
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_furiten(&self) -> bool {
        self.at_furiten
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tiles_seen(&self) -> [u8; 34] { self.tiles_seen }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn last_tedashis2(&self) -> [u8; 4] { self.last_tedashis2 }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn last_tedashis3(&self) -> [[u8; 34]; 3] { self.last_tedashis3 }

    #[getter]
    #[inline]
    #[must_use]
    pub fn last_tedashis4(&self) -> [bool; 34] {
        let mut result: [bool; 34] = [false; 34];

        let mut seen2: [u8; 34] = [0; 34];
        for i in 0..34 {
            seen2[i] = self.tiles_seen[i] - self.tehai[i];
        }

        // 国士
        let yaojiu: [u8; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        let mut guo = true;
        for &index in yaojiu.iter() {
            if seen2[index as usize] == 4 {
                guo = false;
                break;
            }
        }

        // 遍历 tiles_seen 数组，设置 result 数组的值
        for i in 0..34 {
            // 现四枚
            if seen2[i] == 4 {
                result[i] = true;
                continue;
            }
            // 现三枚
            if seen2[i] == 3 {
                // 幺九
                if i == 0 || i == 8 || i == 9 || i == 17 || i == 18 || (26 <= i && i <= 33) {
                    if guo {
                        continue;
                    }
                    // 不能国 幺九看壁外
                    if i == 0 || i == 9 || i == 18 {
                        if self.tiles_seen[i+1] == 4 || self.tiles_seen[i+2] == 4 {
                            result[i] = true;
                            continue;
                        }
                    }
                    else if i == 8 || i == 17 || i == 26 {
                        if self.tiles_seen[i-1] == 4 || self.tiles_seen[i-2] == 4 {
                            result[i] = true;
                            continue;
                        }
                    }
                    else {
                        result[i] = true;
                        continue;
                    }
                } else {
                    // 2m 2p 2s
                    if i == 1 || i == 10 || i == 19 {
                        if self.tiles_seen[i-1] == 4 && (self.tiles_seen[i+1] == 4 || self.tiles_seen[i+2] == 4) {
                            result[i] = true;
                            continue;
                        }
                    }
                    // 8m 8p 8s
                    else if i == 7 || i == 16 || i == 25 {
                        if self.tiles_seen[i+1] == 4 && (self.tiles_seen[i-1] == 4 || self.tiles_seen[i-2] == 4) {
                            result[i] = true;
                            continue;
                        }
                    }
                    else {
                        if (self.tiles_seen[i-1] == 4 && self.tiles_seen[i+1] == 4) || (self.tiles_seen[i-2] == 4 && self.tiles_seen[i+1] == 4) || (self.tiles_seen[i-1] == 4 && self.tiles_seen[i+2] == 4) {
                            result[i] = true;
                            continue;
                        }
                    }
                }
            }
        }
        return result;
    }

    #[getter]
    #[inline]
    #[must_use]
    pub fn last_tedashis5(&self) -> [bool; 34] {
        let mut result: [bool; 34] = [false; 34];

        let mut seen2: [u8; 34] = [0; 34];
        for i in 0..34 {
            seen2[i] = self.tiles_seen[i] - self.tehai[i];
        }

        // 遍历 tiles_seen 数组，设置 result 数组的值
        for i in 0..34 {
            // 现四枚
            if seen2[i] == 4 {
                result[i] = true;
                continue;
            }
            // 现三枚
            if seen2[i] == 3 {
                // 幺九
                if i == 0 || i == 8 || i == 9 || i == 17 || i == 18 || (26 <= i && i <= 33) {
                    // 不能国 幺九看壁外
                    if i == 0 || i == 9 || i == 18 {
                        if self.tiles_seen[i+1] == 4 || self.tiles_seen[i+2] == 4 {
                            result[i] = true;
                            continue;
                        }
                    }
                    else if i == 8 || i == 17 || i == 26 {
                        if self.tiles_seen[i-1] == 4 || self.tiles_seen[i-2] == 4 {
                            result[i] = true;
                            continue;
                        }
                    }
                    else {
                        result[i] = true;
                        continue;
                    }
                } else {
                    // 2m 2p 2s
                    if i == 1 || i == 10 || i == 19 {
                        if self.tiles_seen[i-1] == 4 && (self.tiles_seen[i+1] == 4 || self.tiles_seen[i+2] == 4) {
                            result[i] = true;
                            continue;
                        }
                    }
                    // 8m 8p 8s
                    else if i == 7 || i == 16 || i == 25 {
                        if self.tiles_seen[i+1] == 4 && (self.tiles_seen[i-1] == 4 || self.tiles_seen[i-2] == 4) {
                            result[i] = true;
                            continue;
                        }
                    }
                    else {
                        if (self.tiles_seen[i-1] == 4 && self.tiles_seen[i+1] == 4) || (self.tiles_seen[i-2] == 4 && self.tiles_seen[i+1] == 4) || (self.tiles_seen[i-1] == 4 && self.tiles_seen[i+2] == 4) {
                            result[i] = true;
                            continue;
                        }
                    }
                }
            }
        }
        return result;
    }
}

impl PlayerState {
    #[inline]
    #[must_use]
    pub const fn last_self_tsumo(&self) -> Option<Tile> {
        self.last_self_tsumo
    }
    #[inline]
    #[must_use]
    pub const fn last_kawa_tile(&self) -> Option<Tile> {
        self.last_kawa_tile
    }

    #[inline]
    #[must_use]
    pub fn ankan_candidates(&self) -> &[Tile] {
        &self.ankan_candidates
    }
    #[inline]
    #[must_use]
    pub fn kakan_candidates(&self) -> &[Tile] {
        &self.kakan_candidates
    }
}
