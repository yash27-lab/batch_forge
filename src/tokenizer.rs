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
    #[error("unknown tokenizer ID: {0}")]
    UnknownTokenId(usize),
    #[error("duplicate BPE merge pair: {0:?}")]
    DuplicateMerge(String),
    #[error("invalid BPE merge line: {0:?}")]
    InvalidMergeLine(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse vocab.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("vocab.json assigns token ID {0} to more than one token")]
    DuplicateTokenId(usize),
    #[error("vocab.json is missing token ID {0} from its dense range")]
    MissingTokenId(usize),
    #[error("vocab.json is missing tokenizer symbol {0:?}")]
    MissingVocabSymbol(String),
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
    /// Loads a tokenizer from vocab.json and merges.txt.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, TokenizerError> {
        let vocab_raw = fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&vocab_raw)?;
        let merges_raw = fs::read_to_string(merges_path)?;
        Self::from_assets(vocab, &merges_raw)
    }

    fn from_assets(
        vocab: HashMap<String, usize>,
        merges_raw: &str,
    ) -> Result<Self, TokenizerError> {
        let mut decoder = HashMap::with_capacity(vocab.len());
        for (token, &id) in &vocab {
            if decoder.insert(id, token.clone()).is_some() {
                return Err(TokenizerError::DuplicateTokenId(id));
            }
        }
        for id in 0..vocab.len() {
            if !decoder.contains_key(&id) {
                return Err(TokenizerError::MissingTokenId(id));
            }
        }

        let byte_encoder = bytes_to_unicode();
        for &symbol in &byte_encoder {
            let symbol = symbol.to_string();
            if !vocab.contains_key(&symbol) {
                return Err(TokenizerError::MissingVocabSymbol(symbol));
            }
        }

        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_raw
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("#version:"))
            .enumerate()
        {
            let mut it = line.split_whitespace();
            if let (Some(a), Some(b), None) = (it.next(), it.next(), it.next()) {
                for operand in [a, b] {
                    if !vocab.contains_key(operand) {
                        return Err(TokenizerError::MissingVocabSymbol(operand.to_string()));
                    }
                }
                let merged = format!("{a}{b}");
                if !vocab.contains_key(&merged) {
                    return Err(TokenizerError::MissingVocabSymbol(merged));
                }
                if bpe_ranks
                    .insert((a.to_string(), b.to_string()), rank)
                    .is_some()
                {
                    return Err(TokenizerError::DuplicateMerge(line.to_string()));
                }
            } else {
                return Err(TokenizerError::InvalidMergeLine(line.to_string()));
            }
        }

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

    /// Decodes token IDs, returning an error instead of skipping unknown IDs.
    pub fn try_decode(&self, ids: &[usize]) -> Result<String, TokenizerError> {
        Ok(String::from_utf8_lossy(&self.try_decode_bytes(ids)?).into_owned())
    }

    /// Decodes raw bytes while checking every token ID.
    pub fn try_decode_bytes(&self, ids: &[usize]) -> Result<Vec<u8>, TokenizerError> {
        let mut bytes = Vec::new();
        for id in ids {
            let token = self
                .decoder
                .get(id)
                .ok_or(TokenizerError::UnknownTokenId(*id))?;
            bytes.extend(
                token
                    .chars()
                    .filter_map(|c| self.byte_decoder.get(&c).copied()),
            );
        }
        Ok(bytes)
    }

    /// Decodes token ids back into text.
    pub fn decode(&self, ids: &[usize]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Decodes raw bytes, allowing streaming callers to retain partial UTF-8 sequences.
    pub fn decode_bytes(&self, ids: &[usize]) -> Vec<u8> {
        let mapped: String = ids
            .iter()
            .filter_map(|id| self.decoder.get(id))
            .flat_map(|s| s.chars())
            .collect();
        mapped
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect()
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
            if e < n && e - i > 1 {
                e -= 1;
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

    fn byte_vocab() -> HashMap<String, usize> {
        bytes_to_unicode()
            .iter()
            .enumerate()
            .map(|(id, symbol)| (symbol.to_string(), id))
            .collect()
    }

    #[test]
    fn rejects_duplicate_token_ids() {
        let vocab = HashMap::from([("first".to_string(), 0), ("second".to_string(), 0)]);
        assert!(matches!(
            Tokenizer::from_assets(vocab, ""),
            Err(TokenizerError::DuplicateTokenId(0))
        ));
    }

    #[test]
    fn rejects_sparse_token_ids() {
        let vocab = HashMap::from([("first".to_string(), 0), ("third".to_string(), 2)]);
        assert!(matches!(
            Tokenizer::from_assets(vocab, ""),
            Err(TokenizerError::MissingTokenId(1))
        ));
    }

    #[test]
    fn rejects_vocab_missing_a_byte_symbol() {
        let vocab = HashMap::from([("a".to_string(), 0)]);
        assert!(matches!(
            Tokenizer::from_assets(vocab, ""),
            Err(TokenizerError::MissingVocabSymbol(_))
        ));
    }

    #[test]
    fn rejects_merge_results_missing_from_vocab() {
        let vocab = byte_vocab();
        assert!(matches!(
            Tokenizer::from_assets(vocab, "#version: 0.2\na b\n"),
            Err(TokenizerError::MissingVocabSymbol(symbol)) if symbol == "ab"
        ));
    }

    #[test]
    fn valid_byte_vocab_encodes_and_decodes_a_merge() {
        let mut vocab = byte_vocab();
        vocab.insert("ab".to_string(), 256);
        let tokenizer = Tokenizer::from_assets(vocab, "#version: 0.2\na b\n").unwrap();
        assert_eq!(tokenizer.encode("ab"), vec![256]);
        assert_eq!(tokenizer.decode(&[256]), "ab");
    }

    #[test]
    fn whitespace_runs_match_gpt2_lookahead() {
        assert_eq!(pre_tokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pre_tokenize("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(pre_tokenize("a  "), vec!["a", "  "]);
        assert_eq!(pre_tokenize("a\n b"), vec!["a", "\n", " b"]);
    }

    #[test]
    fn hash_prefix_merges_are_not_comments() {
        let mut vocab = byte_vocab();
        vocab.insert("##".into(), 256);
        let tokenizer = Tokenizer::from_assets(vocab, "#version: 0.2\n\n# #\n").unwrap();
        assert_eq!(tokenizer.encode("##"), vec![256]);
    }

    #[test]
    fn malformed_merge_lines_are_reported() {
        for line in ["a", "a b c"] {
            assert!(matches!(
                Tokenizer::from_assets(byte_vocab(), line),
                Err(TokenizerError::InvalidMergeLine(_))
            ));
        }
    }

    #[test]
    fn duplicate_merges_are_rejected() {
        let mut vocab = byte_vocab();
        vocab.insert("ab".into(), 256);
        assert!(matches!(
            Tokenizer::from_assets(vocab, "a b\na b\n"),
            Err(TokenizerError::DuplicateMerge(_))
        ));
    }

    #[test]
    fn missing_merge_operands_are_reported() {
        let mut vocab = byte_vocab();
        vocab.insert("not_presenta".into(), 256);
        assert!(
            matches!(Tokenizer::from_assets(vocab, "not_present a"), Err(TokenizerError::MissingVocabSymbol(symbol)) if symbol == "not_present")
        );
    }

    #[test]
    fn checked_decode_reports_unknown_ids_and_preserves_bytes() {
        let tokenizer = Tokenizer::from_assets(byte_vocab(), "").unwrap();
        assert!(matches!(
            tokenizer.try_decode(&[999]),
            Err(TokenizerError::UnknownTokenId(999))
        ));
        assert_eq!(
            tokenizer.try_decode_bytes(&[0xe2, 0x82, 0xac]).unwrap(),
            vec![0xe2, 0x82, 0xac]
        );
        assert_eq!(tokenizer.try_decode(&[0xe2, 0x82, 0xac]).unwrap(), "€");
    }
}
