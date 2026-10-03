//! GPT-2 byte-level BPE tokenizer, implemented from scratch.
//!
//! Mirrors the original GPT-2 encoder: bytes are mapped into a printable
//! unicode alphabet, text is pre-tokenized into word-like chunks, and byte-pair
//! merges are applied in rank order from `merges.txt`. Loads the same
//! `vocab.json` / `merges.txt` HuggingFace ships for `gpt2`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse vocab.json: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct Tokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

/// GPT-2's reversible bytes→unicode mapping (every byte gets a printable char).
fn bytes_to_unicode() -> [char; 256] {
    let mut in_set = [false; 256];
    let mut cp = [0u32; 256];
    let push_range = |a: u32, b: u32, in_set: &mut [bool; 256], cp: &mut [u32; 256]| {
        for c in a..=b {
            in_set[c as usize] = true;
            cp[c as usize] = c;
        }
    };
    push_range(b'!' as u32, b'~' as u32, &mut in_set, &mut cp);
    push_range(0xA1, 0xAC, &mut in_set, &mut cp);
    push_range(0xAE, 0xFF, &mut in_set, &mut cp);
    let mut n = 0u32;
    for b in 0..256usize {
        if !in_set[b] {
            cp[b] = 256 + n;
            n += 1;
        }
    }
    let mut arr = ['\0'; 256];
    for b in 0..256usize {
        arr[b] = char::from_u32(cp[b]).unwrap();
    }
    arr
}

impl Tokenizer {
    /// Loads a tokenizer from `vocab.json` and `merges.txt`.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, TokenizerError> {
        let vocab_raw = fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&vocab_raw)?;
        let decoder: HashMap<usize, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        let merges_raw = fs::read_to_string(merges_path)?;
        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_raw
            .lines()
            .filter(|l| !l.starts_with('#'))
            .enumerate()
        {
            let mut it = line.split_whitespace();
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                bpe_ranks.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let byte_encoder = bytes_to_unicode();
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        Ok(Self {
            encoder: vocab,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Encodes text into token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for chunk in pre_tokenize(text) {
            // Map each UTF-8 byte of the chunk into the unicode alphabet.
            let mapped: String = chunk
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            for sym in self.bpe(&mapped) {
                if let Some(&id) = self.encoder.get(&sym) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Decodes token ids back into text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mapped: String = ids
            .iter()
            .filter_map(|id| self.decoder.get(id))
            .flat_map(|s| s.chars())
            .collect();
        let bytes: Vec<u8> = mapped
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Applies BPE merges to one pre-tokenized, byte-mapped chunk.
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() < 2 {
            return word;
        }
        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, (String, String))> = None;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if best.as_ref().map_or(true, |(r, _)| rank < *r) {
                        best = Some((rank, pair));
                    }
                }
            }
            let Some((_, pair)) = best else { break };
            // Merge every non-overlapping occurrence of that pair.
            let merged = format!("{}{}", pair.0, pair.1);
            let mut next = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(word[i].clone());
                    i += 1;
                }
            }
            word = next;
        }
        word
    }
}

/// Splits text into GPT-2-style chunks (contractions, words with an optional
/// leading space, number runs, punctuation runs, and whitespace runs). This is
/// a hand-rolled equivalent of GPT-2's regex that covers ordinary prose.
fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];

        // Contractions: 's 't 're 've 'm 'll 'd
        if c == '\'' && i + 1 < n {
            let two: String = chars[i + 1..n.min(i + 3)].iter().collect();
            if ["re", "ve", "ll"].contains(&two.as_str()) {
                out.push(format!("'{two}"));
                i += 3;
                continue;
            }
            let one = chars[i + 1];
            if matches!(one, 's' | 't' | 'm' | 'd') {
                out.push(format!("'{one}"));
                i += 2;
                continue;
            }
        }

        // Optional single leading space attached to the following word/number/punct.
        let lead_space = c == ' ' && i + 1 < n && !chars[i + 1].is_whitespace();
        let start = i;
        let k = if lead_space { i + 1 } else { i };
        if k < n && !chars[k].is_whitespace() {
            let cat = category(chars[k]);
            let mut e = k + 1;
            while e < n && category(chars[e]) == cat && !chars[e].is_whitespace() {
                e += 1;
            }
            out.push(chars[start..e].iter().collect());
            i = e;
            continue;
        }

        // Whitespace run.
        if c.is_whitespace() {
            let mut e = i;
            while e < n && chars[e].is_whitespace() {
                e += 1;
            }
            out.push(chars[i..e].iter().collect());
            i = e;
            continue;
        }

        out.push(c.to_string());
        i += 1;
    }
    out
}

#[derive(PartialEq, Eq)]
enum Cat {
    Letter,
    Digit,
    Other,
}

fn category(c: char) -> Cat {
    if c.is_alphabetic() {
        Cat::Letter
    } else if c.is_numeric() {
        Cat::Digit
    } else {
        Cat::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maybe_tokenizer() -> Option<Tokenizer> {
        let v = Path::new("models/gpt2/vocab.json");
        let m = Path::new("models/gpt2/merges.txt");
        if v.exists() && m.exists() {
            Tokenizer::from_files(v, m).ok()
        } else {
            None
        }
    }

    #[test]
    fn encodes_known_example() {
        let Some(tok) = maybe_tokenizer() else {
            eprintln!("skipping: GPT-2 tokenizer files not present");
            return;
        };
        // Canonical GPT-2 encoding.
        assert_eq!(tok.encode("Hello world"), vec![15496, 995]);
    }

    #[test]
    fn roundtrips() {
        let Some(tok) = maybe_tokenizer() else { return };
        for s in [
            "The quick brown fox.",
            "batch_forge runs GPT-2!",
            "I'm here",
        ] {
            assert_eq!(tok.decode(&tok.encode(s)), s);
        }
    }
}
