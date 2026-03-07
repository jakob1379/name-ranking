# Pronounceability & Articulatory Features for Name Preference Ranking

## Research Summary for Bradley-Terry Bayesian Model

Based on computational phonology research, psycholinguistic studies, and Danish
phonology. Features designed for spelling-based approximations with IPA
transcription when available.

---

## 1. Sonority Sequencing Compliance

**Feature:** `sonority_sequencing_score`

**Computation:**

```python
# Sonority hierarchy (high to low):
# Vowels > Glides > Liquids > Nasals > Fricatives > Stops

sonority_scores = {
    # Vowels (high sonority - 5)
    'a': 5, 'e': 5, 'i': 5, 'o': 5, 'u': 5, 'y': 5, 'æ': 5, 'ø': 5, 'å': 5,
    # Glides (4)
    'j': 4, 'w': 4,
    # Liquids (3)
    'l': 3, 'r': 3,
    # Nasals (2)
    'm': 2, 'n': 2, 'ŋ': 2,
    # Fricatives (1)
    'f': 1, 'v': 1, 's': 1, 'z': 1, 'ʃ': 1, 'ʒ': 1, 'h': 1, 'ð': 1,
    # Stops (0)
    'p': 0, 'b': 0, 't': 0, 'd': 0, 'k': 0, 'g': 0
}

def calculate_ssp_score(phonemes):
    """
    Sonority should peak at the nucleus and decrease toward edges.
    Violations = sonority increases away from the peak when it should decrease.
    """
    # Find sonority peak (usually vowel)
    sonorities = [sonority_scores.get(p, 0) for p in phonemes]
    peak_idx = sonorities.index(max(sonorities))

    violations = 0
    # Check left side (should decrease toward onset)
    for i in range(peak_idx):
        if sonorities[i] > sonorities[i+1]:
            violations += 1
    # Check right side (should decrease toward coda)
    for i in range(peak_idx, len(sonorities)-1):
        if sonorities[i] < sonorities[i+1]:
            violations += 1

    return max(0, 1 - (violations / len(phonemes)))
```

**Expected Correlation:** **Positive (strong)**

- Names with natural sonority profiles feel easier to say
- Research: Jespersen (1904), Selkirk (1984), Bartlett et al. (2009)
- High SSP compliance = smoother articulation = higher preference

**Danish Considerations:**

- Stød names (e.g., "Morten", "Kirsten") have unique sonority patterns
- Danish has frequent sonority violations in loan names

---

## 2. Articulatory Complexity Index

**Feature:** `articulatory_complexity`

**Computation:**

```python
# Assign articulatory effort scores based on phoneme features
articulatory_effort = {
    # Consonants by place/manner
    'p': 0.3, 'b': 0.3,  # Bilabial - low effort
    't': 0.4, 'd': 0.4,  # Alveolar - low effort
    'k': 0.5, 'g': 0.5,  # Velar - medium effort
    'f': 0.2, 'v': 0.2,  # Labiodental - low effort
    's': 0.3, 'z': 0.3,  # Alveolar fricative - low effort
    'ʃ': 0.6, 'ʒ': 0.6,  # Postalveolar - higher effort
    'h': 0.1,            # Glottal - very low effort
    'ð': 0.7,            # Dental fricative - Danish-specific, higher effort
    'l': 0.3, 'r': 0.5,  # Liquids (Danish r is uvular/velar)
    'm': 0.2, 'n': 0.2, 'ŋ': 0.4,  # Nasals
    'j': 0.2, 'w': 0.2,  # Glides - low effort
    # Vowels by tongue position complexity
    'i': 0.2, 'u': 0.2,  # High vowels - simple
    'e': 0.3, 'o': 0.3,  # Mid vowels
    'a': 0.3,            # Low vowel
    'æ': 0.4, 'ø': 0.5, 'å': 0.4,  # Danish-specific
    'y': 0.4             # Rounded front - moderate effort
}

def articulatory_complexity(phonemes):
    """Sum of individual phoneme effort + transition penalties"""
    base_effort = sum(articulatory_effort.get(p, 0.5) for p in phonemes)

    # Transition penalties for articulatory gymnastics
    transition_penalty = 0
    for i in range(len(phonemes) - 1):
        p1, p2 = phonemes[i], phonemes[i+1]
        # Large place of articulation jumps
        if is_large_transition(p1, p2):
            transition_penalty += 0.3

    # Normalize by length
    return (base_effort + transition_penalty) / len(phonemes)
```

**Expected Correlation:** **Negative (strong)**

- Lower complexity = easier articulation = higher preference
- Research: Articulatory Phonology framework (Browman & Goldstein 1986-1992)
- Danish-specific: Soft consonants (lenition) actually reduce complexity

**Implementation Note:** Invert this score (1 - normalized) for positive
correlation with preference.

---

## 3. Phoneme Transition Probabilities (Wordlikeness)

**Feature:** `phonotactic_probability` / `wordlikeness_score`

**Computation:**

```python
from collections import defaultdict
import math

class PhonotacticModel:
    """
    N-gram phonotactic probability model trained on Danish name corpus.
    Based on UCI Phonotactic Calculator methodology (Mayer et al. 2025).
    """

    def __init__(self, training_names):
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.unigrams = defaultdict(int)
        self.total = 0
        self._train(training_names)

    def _train(self, names):
        for name in names:
            phonemes = self._to_phonemes(name)
            # Add boundary markers
            phonemes = ['<s>'] + phonemes + ['</s>']

            for p in phonemes[1:-1]:  # Exclude boundaries for unigrams
                self.unigrams[p] += 1
                self.total += 1

            for i in range(len(phonemes) - 1):
                self.bigrams[phonemes[i]][phonemes[i+1]] += 1

    def score(self, name, method='positional'):
        """
        Methods:
        - 'unigram': Average unigram probability (segment probability)
        - 'bigram': Average bigram probability (transition probability)
        - 'positional': Sum of positional probabilities (onset/nucleus/coda)
        """
        phonemes = self._to_phonemes(name)
        phonemes = ['<s>'] + phonemes + ['</s>']

        if method == 'unigram':
            probs = [self._unigram_prob(p) for p in phonemes[1:-1]]
            return sum(math.log(p) for p in probs) / len(probs)

        elif method == 'bigram':
            probs = []
            for i in range(len(phonemes) - 1):
                p = self._bigram_prob(phonemes[i], phonemes[i+1])
                probs.append(math.log(p))
            return sum(probs) / len(probs)

        elif method == 'positional':
            # Weight by syllable position (onset, nucleus, coda)
            return self._positional_score(phonemes)

    def _bigram_prob(self, p1, p2):
        count = self.bigrams[p1][p2]
        total = sum(self.bigrams[p1].values())
        return (count + 1) / (total + len(self.unigrams))  # Add-1 smoothing
```

**Expected Correlation:** **Positive (moderate-strong)**

- Higher phonotactic probability = more "name-like" = higher preference
- Research: Vitevitch & Luce (2004), Storkel & Hoover (2010)
- Wordlikeness correlates with familiarity and processing ease

**Key Metrics:**

- **Positional Bigram Frequency**: P(phoneme | position in word)
- **Biphonic Probability**: Co-occurrence of adjacent phonemes
- **Neighborhood Density**: Number of similar-sounding existing names

---

## 4. Syllable Complexity Analysis

**Feature:** `syllable_complexity_score`

**Computation:**

```python
def analyze_syllable_complexity(name):
    """
    Analyze onset + nucleus + coda structure for each syllable.
    Based on complexity hypothesis (Gierut 2007).
    """
    syllables = syllabify(name)  # Using SSP or pyphen

    complexity_scores = []
    for syl in syllables:
        onset, nucleus, coda = parse_syllable_structure(syl)

        # Complexity components
        onset_score = len(onset) ** 2  # Quadratic penalty for clusters
        coda_score = len(coda) ** 1.5  # Less penalty for codas

        # Markedness adjustments
        if has_complex_cluster(onset):
            onset_score *= 1.5
        if has_obstruent_coda(coda):
            coda_score *= 1.2

        syl_complexity = onset_score + 1 + coda_score  # +1 for nucleus
        complexity_scores.append(syl_complexity)

    # Aggregate: max complexity matters more than average
    return max(complexity_scores) * 0.6 + np.mean(complexity_scores) * 0.4

def parse_syllable_structure(syllable_phonemes):
    """
    Split syllable into onset (pre-vowel), nucleus (vowel), coda (post-vowel).
    """
    vowels = {'a', 'e', 'i', 'o', 'u', 'y', 'æ', 'ø', 'å'}

    nucleus_idx = None
    for i, p in enumerate(syllable_phonemes):
        if p in vowels:
            nucleus_idx = i
            break

    if nucleus_idx is None:
        return syllable_phonemes, [], []  # No nucleus found

    onset = syllable_phonemes[:nucleus_idx]
    nucleus = [syllable_phonemes[nucleus_idx]]
    coda = syllable_phonemes[nucleus_idx+1:]

    return onset, nucleus, coda
```

**Expected Correlation:** **Negative (moderate)**

- Simpler syllable structures preferred
- Research: Complexity Hypothesis (Anttila 2008), Gierut (2007)
- Optimal: CV (consonant-vowel) or CVC structures

**Danish-Specific Patterns:**

- Common: CVC ("Lars", "Mette")
- Moderate: CCVC ("Freja", "Kristian")
- Complex: CCCVCC (rare in Danish names)

---

## 5. Wordlikeness / Nonce Word Acceptability

**Feature:** `wordlikeness_rating`

**Computation:**

```python
def compute_wordlikeness(name, training_corpus):
    """
    Multi-component wordlikeness score.
    Based on Bailey & Hahn (2001), Vitevitch & Luce (1998).
    """
    phonemes = to_phonemes(name)

    # Component 1: Positional phoneme probability
    pos_prob = sum(
        log_positional_prob(phonemes[i], i, len(phonemes))
        for i in range(len(phonemes))
    ) / len(phonemes)

    # Component 2: Average bigram probability
    bigram_prob = compute_avg_bigram_prob(phonemes, training_corpus)

    # Component 3: Neighborhood density
    density = phonological_neighborhood_density(name, training_corpus)
    density_score = min(density / 10, 1.0)  # Cap at 10 neighbors

    # Component 4: Sonority profile similarity to real names
    sonority_profile = compute_sonority_profile(phonemes)
    typical_profile = get_typical_sonority_profile(training_corpus)
    profile_match = 1 - jsd(sonority_profile, typical_profile)  # Jensen-Shannon

    # Weighted combination
    wordlikeness = (
        0.25 * normalize(pos_prob) +
        0.30 * normalize(bigram_prob) +
        0.25 * density_score +
        0.20 * profile_match
    )

    return wordlikeness

def phonological_neighborhood_density(name, corpus, threshold=1):
    """
    Count names differing by 1 phoneme (Levenshtein distance at phoneme level).
    """
    name_phonemes = to_phonemes(name)
    neighbors = 0

    for other in corpus:
        other_phonemes = to_phonemes(other)
        if phoneme_levenshtein(name_phonemes, other_phonemes) <= threshold:
            neighbors += 1

    return neighbors
```

**Expected Correlation:** **Positive (strong)**

- Names that "sound like" real names are preferred
- Research: Needle et al. (2019) on pseudoword acceptability
- Wordlikeness captures implicit knowledge of phonotactic constraints

---

## 6. Orthographic Transparency (Grapheme-Phoneme Consistency)

**Feature:** `orthographic_transparency`

**Computation:**

```python
# Danish-specific grapheme-to-phoneme mappings
danish_g2p = {
    'a': ['a', 'ɑ'],
    'e': ['e', 'ɛ', 'ə'],  # Schwa in unstressed positions
    'i': ['i'],
    'o': ['o', 'ɔ'],
    'u': ['u', 'o'],
    'y': ['y'],
    'æ': ['ɛ', 'æ'],
    'ø': ['ø', 'œ'],
    'å': ['ɔ', 'o'],
    'g': ['g', 'j', ''],   # Often silent or palatalized
    'd': ['d', 'ð', ''],   # Soft d (ð) in many positions
    'r': ['ʁ', 'ɐ'],       # Uvular or vocalized
    'v': ['v', 'w'],
    'j': ['j', ''],
    'k': ['k', 'ɡ'],
    'c': ['k', 's'],
    # ... etc
}

def orthographic_depth(spelling):
    """
    Measure spelling-pronunciation consistency.
    Lower score = more transparent (shallow orthography).
    """
    pronunciation = to_phonemes(spelling)

    # Method 1: One-to-many mapping count
    ambiguity_score = 0
    for letter in spelling.lower():
        if letter in danish_g2p:
            ambiguity_score += len(danish_g2p[letter]) - 1

    # Method 2: Regularity of letter-sound correspondence
    regularity = 0
    for i, letter in enumerate(spelling.lower()):
        expected_sounds = danish_g2p.get(letter, [letter])
        actual_sound = pronunciation[i] if i < len(pronunciation) else None

        if actual_sound in expected_sounds:
            regularity += 1

    regularity_ratio = regularity / len(spelling)

    # Combined: lower is more transparent
    return (ambiguity_score / len(spelling)) * 0.5 + (1 - regularity_ratio) * 0.5
```

**Expected Correlation:** **Positive (weak-moderate)**

- More transparent spelling = easier to learn/remember name
- Research: Danish is relatively shallow orthography
- May matter less for familiar names, more for novel names

**Danish Note:** Danish orthography is moderately transparent but has notable
irregularities:

- Silent 'd' ("fader" pronounced with soft ð)
- 'g' often silent or j-like
- 'r' varies by dialect

---

## 7. Syllable Sonority Peaks

**Feature:** `sonority_peak_prominence`

**Computation:**

```python
def sonority_peak_analysis(name):
    """
    Analyze the prominence of sonority peaks across syllables.
    Strong, clear peaks = more natural rhythm.
    """
    syllables = syllabify(name)
    peaks = []

    for syl in syllables:
        phonemes = to_phonemes(syl)
        sonorities = [sonority_scores.get(p, 0) for p in phonemes]

        # Find peak and its prominence
        peak_idx = sonorities.index(max(sonorities))
        peak_height = sonorities[peak_idx]

        # Calculate prominence: peak height relative to neighbors
        left_valley = min(sonorities[:peak_idx]) if peak_idx > 0 else 0
        right_valley = min(sonorities[peak_idx+1:]) if peak_idx < len(sonorities)-1 else 0

        prominence = peak_height - (left_valley + right_valley) / 2
        peaks.append({
            'height': peak_height,
            'prominence': prominence,
            'position': peak_idx / len(phonemes)  # Normalized position
        })

    # Scoring
    avg_prominence = np.mean([p['prominence'] for p in peaks])
    peak_consistency = 1 - np.std([p['height'] for p in peaks])

    return avg_prominence * 0.7 + peak_consistency * 0.3
```

**Expected Correlation:** **Positive (moderate)**

- Clear, prominent sonority peaks create better rhythmic structure
- Research: Sonority cycle theory (Zec 1995, Parker 2002)
- Flat sonority = monotonous, unclear syllable structure

---

## 8. Consonant Cluster Pronounceability

**Feature:** `cluster_pronounceability`

**Computation:**

```python
# Permissible onset clusters in Danish (simplified)
danish_onset_clusters = {
    's': {'t', 'k', 'p', 'l', 'n', 'm', 'v', 'j', 'r'},  # s-initial
    't': {'r', 'j', 'v'},  # t-initial
    'k': {'r', 'l', 'v', 'j'},  # k-initial
    'p': {'r', 'l', 'j'},  # p-initial
    'b': {'l', 'r', 'j'},
    'd': {'r', 'j'},
    'g': {'l', 'r'},
    'f': {'l', 'r', 'j'},
}

# Sonority distance for valid clusters
min_sonority_distance = 2  # Between onset elements

def cluster_pronounceability(name):
    """
    Score based on cluster complexity and cross-linguistic frequency.
    """
    phonemes = to_phonemes(name)

    cluster_scores = []
    i = 0
    while i < len(phonemes):
        # Find consonant sequences
        if phonemes[i] not in vowels:
            cluster_start = i
            while i < len(phonemes) and phonemes[i] not in vowels:
                i += 1
            cluster = phonemes[cluster_start:i]

            if len(cluster) > 1:
                score = score_cluster(cluster, cluster_start == 0)  # Is onset?
                cluster_scores.append(score)
        else:
            i += 1

    if not cluster_scores:
        return 1.0  # No clusters = maximally pronounceable

    return np.mean(cluster_scores)

def score_cluster(cluster, is_onset):
    """
    Score a consonant cluster (0-1, higher = more pronounceable).
    """
    if len(cluster) == 1:
        return 1.0

    score = 1.0

    # Check sonority sequencing within cluster
    sonorities = [sonority_scores.get(c, 0) for c in cluster]
    if is_onset:
        # Onsets: sonority should rise
        for i in range(len(sonorities) - 1):
            if sonorities[i] >= sonorities[i+1]:
                score -= 0.15
    else:
        # Codas: sonority should fall
        for i in range(len(sonorities) - 1):
            if sonorities[i] <= sonorities[i+1]:
                score -= 0.15

    # Check against language-specific patterns
    if is_onset and len(cluster) == 2:
        c1, c2 = cluster[0], cluster[1]
        if c1 in danish_onset_clusters and c2 in danish_onset_clusters[c1]:
            score += 0.1  # Attested cluster bonus
        else:
            score -= 0.2  # Unattested penalty

    # Length penalty
    if len(cluster) > 2:
        score -= (len(cluster) - 2) * 0.15

    return max(0, min(1, score))
```

**Expected Correlation:** **Positive (moderate)**

- Easier clusters = higher preference
- Research: Consonant cluster acquisition (Barlow 2005)
- Danish allows: st-, sk-, sp-, tr-, kr-, kl-, pr-, pl-, fr-, fl-, br-, bl-,
  etc.

**Common Danish Onsets:**

- Simple: All single consonants
- Two-consonant: st, sk, sp, tr, dr, kr, gr, pr, br, fr, fl, bl, sl, sm, sn
- Three-consonant: Rare ("spr", "skr", "str")

---

## 9. Vowel Harmony / Co-occurrence Patterns

**Feature:** `vowel_coherence`

**Computation:**

```python
def vowel_harmony_score(name):
    """
    Measure vowel consistency across syllables.
    Based on vowel harmony detection methods (Sanders & Knowles).
    """
    vowels_in_name = extract_vowels(name)

    if len(vowels_in_name) <= 1:
        return 1.0  # Single vowel = maximally coherent

    # Feature-based coherence
    vowel_features = {
        'a':  {'back': 1, 'round': 0, 'high': 0},
        'e':  {'back': 0, 'round': 0, 'high': 0},
        'i':  {'back': 0, 'round': 0, 'high': 1},
        'o':  {'back': 1, 'round': 1, 'high': 0},
        'u':  {'back': 1, 'round': 1, 'high': 1},
        'y':  {'back': 0, 'round': 1, 'high': 1},  # Front rounded
        'æ':  {'back': 0, 'round': 0, 'high': 0},
        'ø':  {'back': 0, 'round': 1, 'high': 0},  # Front rounded mid
        'å':  {'back': 1, 'round': 1, 'high': 0},
    }

    # Calculate feature consistency
    feature_variance = {}
    for feature in ['back', 'round', 'high']:
        values = [vowel_features[v][feature] for v in vowels_in_name if v in vowel_features]
        if values:
            feature_variance[feature] = np.var(values)

    # Lower variance = higher coherence
    avg_variance = np.mean(list(feature_variance.values()))
    coherence = 1 - min(avg_variance * 2, 1)  # Scale to 0-1

    # Penalty for radical vowel shifts
    for i in range(len(vowels_in_name) - 1):
        v1, v2 = vowels_in_name[i], vowels_in_name[i+1]
        if v1 in vowel_features and v2 in vowel_features:
            distance = feature_distance(vowel_features[v1], vowel_features[v2])
            if distance > 2:  # Large jump
                coherence -= 0.1

    return max(0, coherence)

def feature_distance(f1, f2):
    """Hamming distance between feature vectors."""
    return sum(f1[k] != f2[k] for k in f1)
```

**Expected Correlation:** **Positive (weak-moderate)**

- Vowel harmony creates cohesive, pleasant sound
- Research: Vowel harmony acquisition (Caplan & Kodner 2017)
- Danish doesn't have strict vowel harmony, but vowel coherence matters

**Note:** Full vowel harmony is rare in Indo-European languages. This measures
vowel consistency rather than strict harmony.

---

## 10. Tongue Twister Detection (Repetitive Articulation)

**Feature:** `repetition_penalty` / `tongue_twister_score`

**Computation:**

```python
def tongue_twister_score(name):
    """
    Detect rapid sound repetition that causes articulatory difficulty.
    Based on computational tongue twister generation research (Keh et al. 2023).
    """
    phonemes = to_phonemes(name)

    scores = []

    # 1. Alliteration (same initial sound repeated)
    if len(phonemes) >= 3:
        alliteration_score = check_alliteration(phonemes)
        scores.append(alliteration_score * 0.3)  # Moderate penalty

    # 2. Rapid place of articulation switches
    place_switches = count_place_switches(phonemes)
    scores.append(min(place_switches * 0.15, 0.5))

    # 3. Manner of articulation repetition
    manner_repetition = count_manner_repetition(phonemes)
    scores.append(manner_repetition * 0.25)

    # 4. Phoneme-level bigram repetition (e.g., "p-p", "t-t")
    immediate_repetition = sum(
        1 for i in range(len(phonemes) - 1)
        if phonemes[i] == phonemes[i+1]
    )
    scores.append(immediate_repetition * 0.3)

    # 5. Similar phoneme repetition (e.g., "t-d", "s-z")
    similar_repetition = count_similar_repetition(phonemes)
    scores.append(similar_repetition * 0.2)

    total_penalty = min(sum(scores), 1.0)
    return 1 - total_penalty  # Return pronounceability (higher = better)

def check_alliteration(phonemes):
    """Check for repeated initial consonants across syllables."""
    # Extract onset consonants
    onsets = []
    for syl in syllabify_phonemes(phonemes):
        onset = [p for p in syl if p not in vowels][:2]  # Up to 2 onset consonants
        if onset:
            onsets.append(onset[0])  # Primary onset

    # Count repeats
    repeats = len(onsets) - len(set(onsets))
    return repeats / max(len(onsets), 1)

def count_place_switches(phonemes):
    """Count rapid switches between articulatory places."""
    place_of_articulation = {
        'p': 'bilabial', 'b': 'bilabial', 'm': 'bilabial',
        'f': 'labiodental', 'v': 'labiodental',
        't': 'alveolar', 'd': 'alveolar', 'n': 'alveolar', 's': 'alveolar',
        'ʃ': 'postalveolar', 'ʒ': 'postalveolar',
        'k': 'velar', 'g': 'velar', 'ŋ': 'velar',
        'h': 'glottal',
    }

    switches = 0
    for i in range(len(phonemes) - 1):
        p1, p2 = phonemes[i], phonemes[i+1]
        if p1 in place_of_articulation and p2 in place_of_articulation:
            if place_of_articulation[p1] != place_of_articulation[p2]:
                # Check if both are obstruents (harder to switch)
                if is_obstruent(p1) and is_obstruent(p2):
                    switches += 1

    return switches
```

**Expected Correlation:** **Negative (moderate)**

- Tongue twister patterns reduce preference
- Research: Keh et al. (2023), Loakman et al. (2024) on tongue twister phonetics
- Repetitive articulation creates processing difficulty

**Common Tongue Twister Patterns:**

- Alliteration: "Peter Piper", "Sally sells"
- Alternating place: "t-k-t-k" sequences
- Similar phonemes: "s-ʃ-s-ʃ" (s-sh-s-sh)

---

## Implementation Priorities

### High Impact (Implement First):

1. **Sonority Sequencing** - Strong theoretical basis, easy to compute
2. **Articulatory Complexity** - Directly relates to ease of speech
3. **Phoneme Transition Probabilities** - Strong empirical support
4. **Syllable Complexity** - Clear preference for simpler structures

### Medium Impact:

5. **Wordlikeness** - Good predictor, requires training corpus
6. **Consonant Cluster Pronounceability** - Important for Danish specifically
7. **Tongue Twister Detection** - Clear negative impact

### Lower Impact / Optional:

8. **Orthographic Transparency** - May matter more for literacy
9. **Vowel Coherence** - Subtler effect
10. **Syllable Sonority Peaks** - May be redundant with SSP

---

## Danish-Specific Considerations

### Phonological Features:

- **Stød**: Glottal creak in some names (e.g., "Morten", "Kirsten")

  - Marker of "authentic Danish" feel
  - Can be detected via syllable structure patterns

- **Soft D (ð)**: Written 'd' pronounced as voiced dental fricative

  - "fader", "moder" - considered elegant/classic
  - "rød", "sød" - common adjectives in names

- **Lenition**: Consonant weakening (p→b, t→d, k→g intervocalically)
  - Affects articulatory complexity calculations

### Orthographic Patterns:

- Silent letters common (g, d, v in certain positions)
- æ, ø, å as distinct vowels
- 'j' often indicates palatalization

### Cultural Factors:

- Classic Danish names often have stød
- International names may lack stød but gain global recognition
- Biblical names (Maria, Peter) have established phonetic patterns

---

## Integration with Bradley-Terry Model

```python
class NamePreferenceFeatures:
    """
    Feature extractor for name preference ranking.
    """

    def __init__(self, training_names):
        self.phonotactic_model = PhonotacticModel(training_names)
        self.corpus_stats = compute_corpus_statistics(training_names)

    def extract(self, name):
        features = {}
        phonemes = self.to_phonemes(name)

        # Core pronounceability features
        features['ssp_score'] = calculate_ssp_score(phonemes)
        features['articulatory_ease'] = 1 - articulatory_complexity(phonemes)
        features['phonotactic_prob'] = self.phonotactic_model.score(name)
        features['syllable_simplicity'] = 1 / (1 + analyze_syllable_complexity(name))
        features['wordlikeness'] = compute_wordlikeness(name, self.corpus_stats)
        features['cluster_ease'] = cluster_pronounceability(name)
        features['no_tongue_twister'] = tongue_twister_score(name)

        # Danish-specific
        features['has_stød_pattern'] = detect_stod_pattern(name)
        features['orthographic_depth'] = 1 - orthographic_depth(name)

        return features
```

---

## References

1. **Jespersen, O.** (1904). _Lehrbuch der Phonetik_. Leipzig.
2. **Selkirk, E.** (1984). On the major class features and syllable theory. In
   _Language Sound Structure_.
3. **Bartlett et al.** (2009). On the syllabification of phonemes. _NAACL_.
4. **Browman & Goldstein** (1986-1992). Articulatory Phonology series. _Haskins
   Labs_.
5. **Mayer et al.** (2025). The UCI Phonotactic Calculator. _Behavior Research
   Methods_.
6. **Vitevitch & Luce** (2004). A web-based interface for calculating
   phonotactic neighborhood. _BRM_.
7. **Anttila** (2008). Gradient phonotactics and the Complexity Hypothesis.
   _NLLT_.
8. **Gierut** (2007). Phonological complexity and language learnability.
   _AJSLP_.
9. **Keh et al.** (2023). PANCETTA: Phoneme Aware Neural Completion for Tongue
   Twisters. _EACL_.
10. **Sidhu & Pexman** (2019). The Sound Symbolism of Names. _Current
    Directions_.
11. **Slepian & Galinsky** (2016). The voiced pronunciation of initial phonemes
    predicts gender. _JPSP_.
12. **Caplan & Kodner** (2017). Vowel Harmony as a Distributional Learning
    Problem. _CogSci_.
13. **Bailey & Hahn** (2001). Determinants of wordlikeness. _Journal of Memory
    and Language_.

---

## Recommended Python Libraries

```python
# Core phonetics/phonology
nltk.tokenize.sonority_sequencing  # SSP implementation
pyphen  # Syllabification
doublemetaphone  # Already using

# Phoneme manipulation
ipa  # IPA handling
epitran  # Grapheme-to-phoneme (supports Danish)

# For training phonotactic models
uci-phonotactic-calculator  # Reference implementation

# Statistics
scipy.stats  # For distributions, entropy
numpy  # General computation
```

---

## Quick Start: Danish-Optimized Feature Set

For immediate implementation with highest ROI:

```python
HIGH_PRIORITY_FEATURES = [
    'ssp_compliance',           # Sonority sequencing
    'articulatory_simplicity',  # Inverse of complexity
    'phonotactic_probability',  # Wordlikeness via n-grams
    'simple_syllable_structure', # CV/CVC preference
    'cluster_naturalness',      # Valid Danish clusters
    'no_twister_pattern',       # No rapid repetition
]

MEDIUM_PRIORITY_FEATURES = [
    'neighborhood_density',     # Similar names exist
    'orthographic_regularity',  # Spelling matches sound
    'vowel_consistency',        # Coherent vowel set
]

OPTIONAL_FEATURES = [
    'stød_presence',           # Danish-specific marker
    'peak_prominence',         # Clear syllable peaks
]
```
