# IndicPulse Data Directory

## Structure

```
data/
├── sample_reviews.csv        ← Demo CSV dataset (30 Hinglish comments)
├── training_labels.csv       ← Labeled dataset for MuRIL fine-tuning
└── README.md
```

## CSV Format Requirements

For uploading your own data, your CSV must have at least one of these column names
(auto-detected by the Universal CSV Parser):

| Field       | Accepted Column Names                                    |
|-------------|----------------------------------------------------------|
| **Text**    | text, comment, review, body, content, message            |
| Author      | author, user, username, name, reviewer                   |
| Timestamp   | timestamp, date, created_at, time, published_at          |
| Likes       | like_count, likes, thumbs_up, upvotes, helpful           |
| ID          | comment_id, id, review_id                                |

## Training Data Format

For fine-tuning MuRIL, your CSV must have exactly:
- `text`: The comment text (Hinglish/English/mixed)
- `label`: Integer class (0=Positive, 1=Sarcastic, 2=Neutral, 3=Toxic)

Minimum recommended: 500+ labeled examples per class.
