### Phonetic Similarity with Double Metaphone

The **Double Metaphone** algorithm converts names to phonetic codes that capture
pronunciation similarities across languages. This is used for:

1. **Similarity Search**: Find names that sound similar to a reference name,
   enabling users to compare phonetically related names.
2. **Cluster‑Based Comparison**: Phonetically similar names are compared early
   in the ranking process to resolve preferences within sound‑alike clusters
   efficiently.
3. **Feature Extraction**: Primary and secondary phonetic codes are included as
   categorical features in the Bradley‑Terry model, allowing the model to learn
   preferences for specific phonetic patterns.
4. **Recommendation Engine**: After a user expresses a preference
   (like/dislike), the system can recommend other names with similar phonetic
   codes, accelerating the ranking process.

The phonetic similarity score is computed as:

- **Exact match**: Both primary codes match (score = 1.0)
- **Primary‑secondary match**: Primary of one matches secondary of the other
  (score = 0.8)
- **Partial match**: Codes share prefix or edit distance within threshold (score
  = 0.5–0.7)
- **No match**: Score = 0.0

This multi‑level scoring allows fine‑grained control over how phonetic
similarity influences pair selection and recommendations.
