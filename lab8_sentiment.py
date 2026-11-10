import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ---------------------------------------------------------------
# Lesson 1: Sample data and sentiment classification
# ---------------------------------------------------------------
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]

df = pd.DataFrame({"id": [f"R{i:02}" for i in range(1, 19)],
                   "text": reviews})

# Task 4: main topic for each comment (assigned manually)
df["topic"] = [
    "Insulation", "Portability", "Lid / Cap", "Durability", "Lid / Cap",
    "Cleaning", "Insulation", "Overall", "Insulation", "Cleaning",
    "Portability", "Overall", "Lid / Cap", "Overall", "Insulation",
    "Value", "Portability", "Overall",
]

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text)))

# ---------------------------------------------------------------
# Lesson 2: Dashboard
# ---------------------------------------------------------------
st.set_page_config(page_title="Lab 8 | Feedback", layout="wide")
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only. Automated labels need human review.")

order = ["Positive", "Neutral", "Negative"]
counts = df["sentiment"].value_counts().reindex(order, fill_value=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total comments", len(df))
m2.metric("Positive", int(counts["Positive"]))
m3.metric("Neutral", int(counts["Neutral"]))
m4.metric("Negative", int(counts["Negative"]))

left, right = st.columns(2)

with left:
    st.subheader("Comment counts by sentiment")
    st.bar_chart(counts, horizontal=True)

with right:
    st.subheader("Words used in the comments")
    # Task 2 changes:
    #  - Added "keep" and "keeps" as stop words: they are frequent but only
    #    describe an action, so on their own they tell the reader nothing.
    #  - Removed "drinks" and "drink" from the stop list so they stay visible.
    #    Next to "cold", they make the insulation theme easier to notice.
    stop_words = set(STOPWORDS) | {"bottle", "keep", "keeps"}
    cloud = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", stopwords=stop_words,
                      collocations=False).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption("Bigger words appear more often across the 18 comments. Size "
               "only shows frequency, not sentiment or importance, and a word "
               "loses its context. \"Cold\" can be praise or a complaint "
               "depending on the sentence. Colour and position mean nothing.")

st.subheader("Review the original comments")
st.dataframe(df[["id", "text", "topic", "sentiment", "compound"]],
             use_container_width=True, hide_index=True)
st.caption("A compound score is a model output, not a confidence rating.")

# ---------------------------------------------------------------
# Lesson 3: Assignment answers
# ---------------------------------------------------------------
st.header("Assignment")

st.subheader("1. Run and describe")
st.markdown(
    f"""
- The app shows **{len(df)} comments**: {counts['Positive']} positive,
  {counts['Neutral']} neutral and {counts['Negative']} negative.
- **Positive** is the largest group, so on the surface people seem mostly happy.
- A recurring theme in the word cloud is **keeping drinks cold** ("cold",
  "drinks", "insulation"), along with ease of use ("easy").
- **Mixed-tone comment:** R07 "Keeps drinks cold, though the paint scratches
  easily." It is labelled Positive (0.34), but it contains a real complaint
  about durability. A single label hides that the product team should hear
  about both the insulation and the paint.
"""
)

st.subheader("2. Improve the qualitative view")
st.markdown(
    """
- **Removed:** "keep" / "keeps". They showed up often but only describe an
  action, so they took space without telling anyone what customers care about.
- **Retained:** "drinks". It used to be a stop word. Keeping it next to "cold"
  makes readers notice that temperature is the main thing people talk about,
  instead of seeing "cold" floating alone.
- A caption under the word cloud now explains what word size means and its limits.
"""
)

st.subheader("3. Audit sentiment labels")
audit_ids = ["R02", "R09", "R01", "R14", "R13", "R08"]
audit = df.set_index("id").loc[audit_ids, ["text", "sentiment", "compound"]].reset_index()
audit["my judgement"] = [
    "Reasonable. Clear praise.",
    "Incorrect. This is a complaint, but 'better' pushed the score up.",
    "Incorrect. Keeping water cold all day is praise; VADER missed it.",
    "Reasonable. Factual and plain, neutral fits.",
    "Reasonable. Clearly negative.",
    "Borderline. 'Fine' reads as neutral, but 'nothing special' made it negative.",
]
st.dataframe(audit, use_container_width=True, hide_index=True)

st.markdown(
    "Other wrong labels I noticed: R15 \"does not keep drinks cold\" is labelled "
    "Positive because of \"Nice\", even though the main point is a failure of "
    "the product."
)

negation = "The bottle is not easy to clean."
neg_label, neg_score = label_sentiment(negation)
pos_label, pos_score = label_sentiment("The bottle is easy to clean.")
st.markdown(
    f"""
**Negation test**
- "The bottle is easy to clean." gives **{pos_label} ({pos_score:.3f})**
- "{negation}" gives **{neg_label} ({neg_score:.3f})**

VADER handled this simple negation correctly and flipped the label. It still
failed on R15, where the negation sits beside a separate positive word, so
negation is only reliable in short, simple sentences.
"""
)

st.subheader("4. Compare and communicate")
topic_table = pd.crosstab(df["topic"], df["sentiment"]).reindex(
    columns=order, fill_value=0)
topic_table["Total"] = topic_table.sum(axis=1)
st.dataframe(topic_table, use_container_width=True)
st.markdown(
    "Each topic only has between 1 and 4 comments, so this table is too small "
    "for a fair comparison. One extra comment would change the picture for any "
    "topic. Insulation is also mislabelled twice out of four, which makes its "
    "row look more positive than it really is. It is useful for spotting "
    "themes, not for ranking them."
)

st.markdown("**Summary for the product team**")
st.markdown(
    """
Out of 18 comments, the sentiment model labelled 9 as positive, 3 as neutral
and 6 as negative, so the overall tone looks favourable. The clearest finding
is that insulation matters most to customers. It is the most common theme, and
reading the comments shows it is split: some people praise how cold the bottle
stays, while others say it does not keep drinks cold or expected better for
the price. The lid and cap are the second concern, with reports of breaking
and being hard to open.

The main limitation is that the automated labels are not fully reliable. In a
manual check, about one in three comments was labelled wrongly or borderline,
mostly comments with mixed opinions or phrases like "expected better". With
only 18 synthetic comments, these numbers should not be used for decisions.

For better feedback, we recommend collecting a larger sample from real buyers
and asking them to rate specific parts of the bottle (lid, insulation,
cleaning, price) separately, with an optional comment box. This gives a
clear score per feature and avoids depending only on a model to read tone.
"""
)

st.subheader("Reflection")
st.markdown(
    """
The visuals showed that insulation and the lid drive most of the feedback.
The bar chart could mislead a reader, because it makes the product look
mostly liked even though several "positive" comments are actually
complaints. The word cloud can also mislead since it strips words of their
meaning. To validate the labels, I would have two people label a sample by
hand and compare their answers with the model. To get more representative
feedback, I would sample buyers at different times after purchase, not just
those who choose to leave a review.
"""
)
