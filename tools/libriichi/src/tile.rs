use crate::{matches_tu8, t, tu8};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::str::FromStr;

use ahash::AHashMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MJAI_PAI_STRINGS_LEN: usize = 3 * 9 + 4 + 3 + 3 + 1;
const MJAI_PAI_STRINGS: [&str; MJAI_PAI_STRINGS_LEN] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", // m
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", // p
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", // s
    "E", "S", "W", "N", "P", "F", "C", // z
    "5mr", "5pr", "5sr", // aka
    "?",   // unknown
];
const DISCARD_PRIORITIES: [u8; 38] = [
    6, 5, 4, 3, 2, 3, 4, 5, 6, // m
    6, 5, 4, 3, 2, 3, 4, 5, 6, // p
    6, 5, 4, 3, 2, 3, 4, 5, 6, // s
    7, 7, 7, 7, 7, 7, 7, // z
    1, 1, 1, // aka
    0, // unknown
];

static MJAI_PAI_STRINGS_MAP: Lazy<AHashMap<&'static str, Tile>> = Lazy::new(|| {
    MJAI_PAI_STRINGS
        .iter()
        .enumerate()
        .map(|(id, &s)| (s, Tile::try_from(id).unwrap()))
        .collect()
});

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile(u8);

#[derive(Debug)]
pub enum InvalidTile {
    Number(usize),
    String(String),
}

impl Tile {
    /// # Safety
    /// Calling this method with an out-of-bounds tile ID is undefined behavior.
    #[inline]
    #[must_use]
    pub const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    pub const fn deaka(self) -> Self {
        match self.0 {
            tu8!(5mr) => t!(5m),
            tu8!(5pr) => t!(5p),
            tu8!(5sr) => t!(5s),
            _ => self,
        }
    }

    #[inline]
    #[must_use]
    pub const fn akaize(self) -> Self {
        match self.0 {
            tu8!(5m) => t!(5mr),
            tu8!(5p) => t!(5pr),
            tu8!(5s) => t!(5sr),
            _ => self,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_aka(self) -> bool {
        matches_tu8!(self.0, 5mr | 5pr | 5sr)
    }

    #[inline]
    #[must_use]
    pub const fn is_jihai(self) -> bool {
        matches_tu8!(self.0, E | S | W | N | P | F | C)
    }

    #[inline]
    #[must_use]
    pub const fn is_yaokyuu(self) -> bool {
        matches_tu8!(
            self.0,
            1m | 9m | 1p | 9p | 1s | 9s | E | S | W | N | P | F | C
        )
    }

    #[inline]
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.0 >= tu8!(?)
    }

    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let kind = tile.0 / 9;
        let num = tile.0 % 9;

        if kind < 3 {
            Self(kind * 9 + (num + 1) % 9)
        } else if num < 4 {
            Self(3 * 9 + (num + 1) % 4)
        } else {
            Self(3 * 9 + 4 + (num - 4 + 1) % 3)
        }
    }

    #[inline]
    #[must_use]
    pub const fn prev(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let kind = tile.0 / 9;
        let num = tile.0 % 9;
        if kind < 3 {
            Self(kind * 9 + (num + 9 - 1) % 9)
        } else if num < 4 {
            Self(3 * 9 + (num + 4 - 1) % 4)
        } else {
            Self(3 * 9 + 4 + (num - 4 + 3 - 1) % 3)
        }
    }

    #[inline]
    #[must_use]
    pub const fn augment(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let tid = tile.0;
        let kind = tid / 9;
        let ret = match kind {
            0 => Self(tid + 9),
            1 => Self(tid - 9),
            _ => tile,
        };
        if self.is_aka() {
            ret.akaize()
        } else {
            ret
        }
    }

    /// `Ordering::Equal` iff `self == other`
    #[inline]
    #[must_use]
    pub fn cmp_discard_priority(self, other: Self) -> Ordering {
        let l = self.0 as usize;
        let r = other.0 as usize;
        match DISCARD_PRIORITIES[l].cmp(&DISCARD_PRIORITIES[r]) {
            Ordering::Equal => r.cmp(&l),
            o => o,
        }
    }
}

impl Default for Tile {
    fn default() -> Self {
        t!(?)
    }
}

impl TryFrom<u8> for Tile {
    type Error = InvalidTile;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        Self::try_from(v as usize)
    }
}

impl TryFrom<usize> for Tile {
    type Error = InvalidTile;

    fn try_from(v: usize) -> Result<Self, Self::Error> {
        if v >= MJAI_PAI_STRINGS_LEN {
            Err(InvalidTile::Number(v))
        } else {
            Ok(Self(v as u8))
        }
    }
}

impl FromStr for Tile {
    type Err = InvalidTile;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MJAI_PAI_STRINGS_MAP
            .get(s)
            .copied()
            .ok_or_else(|| InvalidTile::String(s.to_owned()))
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(MJAI_PAI_STRINGS[self.0 as usize])
    }
}

impl<'de> Deserialize<'de> for Tile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tile = String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)?;
        Ok(tile)
    }
}

impl Serialize for Tile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl fmt::Display for InvalidTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("not a valid tile: ")?;
        match self {
            Self::Number(n) => fmt::Display::fmt(n, f),
            Self::String(s) => write!(f, "not a valid tile: \"{s}\""),
        }
    }
}

impl Error for InvalidTile {}

#[cfg(test)]
mod test {
    use ndarray::s;
    use super::*;

    #[test]
    fn convert() {
        "E".parse::<Tile>().unwrap();
        "5mr".parse::<Tile>().unwrap();
        "?".parse::<Tile>().unwrap();
        Tile::try_from(0_u8).unwrap();
        Tile::try_from(36_u8).unwrap();
        Tile::try_from(37_u8).unwrap();
        println!("{:?}", "E".parse::<Tile>().unwrap());
        println!("{:?}", Tile::try_from(0_u8).unwrap());

        "".parse::<Tile>().unwrap_err();
        "0s".parse::<Tile>().unwrap_err();
        "!".parse::<Tile>().unwrap_err();
        Tile::try_from(38_u8).unwrap_err();
        Tile::try_from(u8::MAX).unwrap_err();
        println!("{:?}", "".parse::<Tile>().unwrap_err());
        println!("{:?}", Tile::try_from(u8::MAX).unwrap_err());
    }

    #[test]
    fn next_prev() {
        MJAI_PAI_STRINGS.iter().take(37).for_each(|&s| {
            let tile: Tile = s.parse().unwrap();
            println!("{:?}", tile);
            println!("{:?}", tile.prev().next());
            println!("{:?}", tile.next().prev());
            assert_eq!(tile.prev().next(), tile.deaka());
            assert_eq!(tile.next().prev(), tile.deaka());
        });
    }

    use crate::algo::agari::{self, AgariCalculator};
    use crate::algo::shanten;
    use crate::algo::sp::{InitState, SPCalculator};
    use crate::hand::hand;
    use crate::state::PlayerState;
    use crate::{t, tu8};

    #[test]
    fn shanten1() {
        shanten::ensure_init();
        let tehai = hand("123456789m 123p 11z").unwrap();
        let y = shanten::calc_all(&tehai, 4);
        println!("{:?}", y);
    }

    #[test]
    fn agari1() {
        agari::ensure_init();
        let tehai = hand("123s 9m 9m").unwrap();
        let calc = AgariCalculator {
                tehai: &tehai,
                is_menzen: false,
                chis: &[],
                pons: &tu8![S, W],
                minkans: &[],
                ankans: &tu8![N,],
                bakaze: tu8!(S),
                jikaze: tu8!(N),
                winning_tile: tu8!(9m),
                is_ron: false,
        };
        let y = calc.search_yakus().unwrap();
        println!("{:?}", y);
    }

    #[test]
    fn sp1() {
        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            prefer_riichi: true,
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &[t!(6m)],
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: true,
            calc_tegawari: true,
            calc_shanten_down: true,
        };
        let tehai = hand("3667m 23489p 34688s").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        let can_discard = true;
        let cur_shanten = shanten::calc_all(&tehai, calc.tehai_len_div3);
        let tsumos_left = 12;
        let init_state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let state = init_state.clone();
        let candidates = &calc
                    .calc(state, can_discard, tsumos_left, cur_shanten)
                    .unwrap();
        println!("{:?}", candidates);

        // ---

        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            prefer_riichi: true,
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &[t!(6m)],
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: true,
            calc_tegawari: true,
            calc_shanten_down: true,
        };
        let tehai = hand("45677m 456788p 234s").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        let can_discard = true;
        let cur_shanten = shanten::calc_all(&tehai, calc.tehai_len_div3);
        let tsumos_left = 12;
        let init_state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let state = init_state.clone();
        let candidates = &calc
                    .calc(state, can_discard, tsumos_left, cur_shanten)
                    .unwrap();
        println!("{:?}", candidates);
    }

    #[test]
    fn encode_obs1() {
        let log = r#"
            {"type":"start_kyoku","bakaze":"S","dora_marker":"F","kyoku":1,"honba":2,"kyotaku":1,"oya":0,"scores":[32300,18000,22000,26700],"tehais":[["1m","3m","4m","5m","6m","6m","7m","5pr","8p","5sr","E","W","N"],["4m","5m","5m","8m","7p","1s","3s","4s","7s","9s","E","E","N"],["1m","2m","3m","3m","7m","9m","6p","9p","1s","4s","5s","8s","C"],["8m","8m","3p","4p","4p","5p","8p","1s","6s","9s","9s","S","F"]]}
            {"type":"tsumo","actor":0,"pai":"C"}
        "#;
        let mut ps = PlayerState::new(0);
        for line in log.trim().split('\n') {
            let x = serde_json::from_str(line).unwrap();
            println!("{:?}", x);
            ps.update(&x);
        }
        let ps = &ps;
        let result = ps.encode_obs(4, false);
        let r2 = result.0;

        use ndarray::{Array2, Axis, NdIndex};
        println!("{:?}", r2.slice(s![..7, ..]));
        println!("{:?}", result.1);
    }
}
