import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).parent
ITEMS_FILE = APP_DIR / "items.csv"
INTERACTIONS_FILE = APP_DIR / "interactions_train.csv"
RECS_FILE = APP_DIR / "recommendations.csv"

st.set_page_config(
    page_title="Uni Library Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root { --card-bg: #171717; --soft: #262626; --accent: #e50914; }
.stApp { background: linear-gradient(135deg, #080808 0%, #181818 55%, #0e0e0e 100%); color: #f5f5f5; }
.main-title { font-size: 3rem; font-weight: 900; letter-spacing: -0.04em; margin-bottom: 0.1rem; }
.subtitle { color: #cfcfcf; font-size: 1.15rem; margin-bottom: 1.2rem; }
.hero { padding: 2rem; border-radius: 28px; background: linear-gradient(120deg, rgba(229,9,20,.25), rgba(255,255,255,.06)); border: 1px solid rgba(255,255,255,.08); margin-bottom: 1rem; }
.metric-card { padding: 1rem; background: rgba(255,255,255,.06); border-radius: 20px; border: 1px solid rgba(255,255,255,.07); }
.book-card { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.08); border-radius: 22px; padding: 1rem; min-height: 620px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
.book-title { font-weight: 800; font-size: 1.03rem; line-height: 1.22; margin-top: .65rem; }
.book-author { color: #d7d7d7; font-size: .9rem; margin-top: .25rem; }
.book-meta { color: #b5b5b5; font-size: .82rem; margin-top: .4rem; }
.reason { color: #ffffff; background: rgba(229,9,20,.20); border-radius: 14px; padding: .55rem; font-size: .82rem; margin-top: .65rem; border: 1px solid rgba(229,9,20,.25); }
.avail { display: inline-block; padding: .25rem .55rem; border-radius: 99px; background: rgba(46, 204, 113, .20); color: #b8ffd5; font-size: .78rem; margin-top: .45rem; }
.no-cover { height: 250px; border-radius: 16px; background: rgba(255,255,255,.08); display:flex; align-items:center; justify-content:center; color:#bdbdbd; text-align:center; padding:1rem; border:1px dashed rgba(255,255,255,.18); }
.small-muted { color:#bdbdbd; font-size:.9rem; }
.stButton>button { border-radius: 999px; font-weight: 700; border: 1px solid rgba(255,255,255,.15); }
section[data-testid="stSidebar"] { background: #111111; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_items() -> pd.DataFrame:
    df = pd.read_csv(ITEMS_FILE)
    df["i"] = pd.to_numeric(df["i"], errors="coerce").astype("Int64")
    for col in ["Title", "Author", "ISBN Valid", "Publisher", "Subjects"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["title_clean"] = df["Title"].str.replace(r"\s*/\s*$", "", regex=True).str.strip()
    df["author_clean"] = df["Author"].replace("nan", "").str.strip()
    df["subject_clean"] = df["Subjects"].str.replace(";", ", ", regex=False).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    df = pd.read_csv(INTERACTIONS_FILE)
    df["u"] = pd.to_numeric(df["u"], errors="coerce").astype("Int64")
    df["i"] = pd.to_numeric(df["i"], errors="coerce").astype("Int64")
    df["borrowed_at"] = pd.to_datetime(df["t"], unit="s", errors="coerce")
    return df.dropna(subset=["u", "i"])


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    # The uploaded recommendation file is named CSV, but its real separator is semicolon.
    try:
        df = pd.read_csv(RECS_FILE, sep=";")
        if "user_id" not in df.columns:
            raise ValueError("fallback")
    except Exception:
        df = pd.read_csv(RECS_FILE)
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(RECS_FILE, sep=";")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["recommendation"] = df["recommendation"].fillna("").astype(str)
    return df.dropna(subset=["user_id"])


def parse_recommendation_ids(value: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(value))]


def first_isbn(isbn_field: str) -> str:
    for part in str(isbn_field).split(";"):
        candidate = re.sub(r"[^0-9Xx]", "", part.strip())
        if len(candidate) in (10, 13):
            return candidate
    return ""


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def openlibrary_cover_url(isbn: str, title: str, author: str) -> str:
    if isbn:
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg?default=false"
    # Real cover lookup by title/author. If no cover is found, return empty string instead of a fake placeholder.
    try:
        params = {"title": title[:120], "author": author[:80], "limit": 1}
        r = requests.get("https://openlibrary.org/search.json", params=params, timeout=2.5)
        if r.ok:
            docs = r.json().get("docs", [])
            if docs and docs[0].get("cover_i"):
                return f"https://covers.openlibrary.org/b/id/{docs[0]['cover_i']}-M.jpg?default=false"
    except Exception:
        return ""
    return ""


def guess_language(row: pd.Series) -> str:
    text = " ".join([row.get("Title", ""), row.get("Subjects", ""), row.get("Publisher", "")]).lower()
    french_markers = ["roman français", "pédagogie", "enseignement", "sciences sociales", "france", "élèves", "langue"]
    english_markers = ["english", "psychology", "economics", "management", "science"]
    german_markers = ["deutsch", "schweiz", "german"]
    if any(m in text for m in french_markers):
        return "French"
    if any(m in text for m in german_markers):
        return "German"
    if any(m in text for m in english_markers):
        return "English"
    return "Any"


def ids_to_books(ids: Iterable[int], items: pd.DataFrame, reason: str, limit: int = 12) -> pd.DataFrame:
    id_list = list(dict.fromkeys(int(x) for x in ids if pd.notna(x)))
    books = items[items["i"].isin(id_list)].copy()
    if books.empty:
        return books
    order = {book_id: pos for pos, book_id in enumerate(id_list)}
    books["sort_order"] = books["i"].map(order)
    books = books.sort_values("sort_order").head(limit)
    books["reason"] = reason
    return books


def popular_books(interactions: pd.DataFrame, items: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    counts = interactions["i"].value_counts().head(300).rename_axis("i").reset_index(name="borrow_count")
    books = counts.merge(items, on="i", how="left").dropna(subset=["Title"]).head(limit)
    books["reason"] = "Popular among library users based on borrowing history."
    return books


def history_for_user(user_id: int, interactions: pd.DataFrame, items: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    hist = interactions[interactions["u"] == user_id].sort_values("borrowed_at", ascending=False).head(limit)
    return hist.merge(items, on="i", how="left")


def same_author_books(history: pd.DataFrame, items: pd.DataFrame, already_seen: set, limit: int = 12) -> pd.DataFrame:
    authors = [a for a in history.get("author_clean", pd.Series(dtype=str)).dropna().unique() if a]
    if not authors:
        return pd.DataFrame()
    books = items[items["author_clean"].isin(authors) & ~items["i"].isin(already_seen)].head(limit).copy()
    books["reason"] = "By an author found in your borrowing history."
    return books


def similar_subject_books(history: pd.DataFrame, items: pd.DataFrame, already_seen: set, limit: int = 12) -> pd.DataFrame:
    subject_text = " ".join(history.get("Subjects", pd.Series(dtype=str)).dropna().astype(str).head(8))
    tokens = [t.lower().strip() for t in re.split(r"[;,]", subject_text) if len(t.strip()) > 4]
    if not tokens:
        return pd.DataFrame()
    pattern = "|".join(re.escape(t) for t in tokens[:8])
    books = items[items["Subjects"].str.lower().str.contains(pattern, na=False) & ~items["i"].isin(already_seen)].head(limit).copy()
    books["reason"] = "Because the subject is similar to books you borrowed."
    return books


def new_user_recommendations(items: pd.DataFrame, interactions: pd.DataFrame, prefs: dict, limit: int = 12) -> pd.DataFrame:
    scored = items.copy()
    scored["score"] = 0
    haystack = (scored["Title"] + " " + scored["Author"] + " " + scored["Subjects"] + " " + scored["Publisher"]).str.lower()

    for term in prefs.get("subjects", []):
        if term:
            scored.loc[haystack.str.contains(re.escape(term.lower()), na=False), "score"] += 4
    for term in prefs.get("authors", []):
        if term:
            scored.loc[scored["Author"].str.lower().str.contains(re.escape(term.lower()), na=False), "score"] += 5
    for term in prefs.get("liked_books", []):
        if term:
            scored.loc[haystack.str.contains(re.escape(term.lower()), na=False), "score"] += 3

    language = prefs.get("language", "Any")
    if language != "Any":
        scored["language_guess"] = scored.apply(guess_language, axis=1)
        scored.loc[scored["language_guess"].isin([language, "Any"]), "score"] += 1

    style = prefs.get("style", "No preference")
    if style == "Academic / study books":
        scored.loc[scored["Subjects"].str.contains("science|sociologie|psychologie|économie|enseignement|méthodologie", case=False, na=False), "score"] += 2
    elif style == "Fiction / novels":
        scored.loc[scored["Subjects"].str.contains("roman|littérature|fiction", case=False, na=False), "score"] += 2
    elif style == "Short and easy reads":
        scored.loc[scored["Title"].str.len() < 70, "score"] += 1

    counts = interactions["i"].value_counts().rename("borrow_count")
    scored = scored.merge(counts, left_on="i", right_index=True, how="left")
    scored["borrow_count"] = scored["borrow_count"].fillna(0)
    scored = scored.sort_values(["score", "borrow_count"], ascending=False)
    result = scored[scored["score"] > 0].head(limit).copy()
    if len(result) < limit:
        fallback = popular_books(interactions, items, limit=limit - len(result))
        result = pd.concat([result, fallback], ignore_index=True)
    result["reason"] = "Matched your onboarding quiz preferences."
    return result.head(limit)


def filter_books(books: pd.DataFrame, query: str, subject_filter: str, available_only: bool) -> pd.DataFrame:
    if books.empty:
        return books
    out = books.copy()
    if query:
        q = query.lower()
        text = (out["Title"] + " " + out["Author"] + " " + out["Subjects"]).str.lower()
        out = out[text.str.contains(re.escape(q), na=False)]
    if subject_filter != "All subjects":
        out = out[out["Subjects"].str.contains(re.escape(subject_filter), case=False, na=False)]
    # Availability is simulated because no inventory/loan-status file was provided.
    if available_only:
        out = out[out["i"].astype(int) % 4 != 0]
    return out


def availability_label(book_id: int) -> str:
    # Transparent demo logic: real availability needs an inventory/current-loans table.
    return "Available now" if int(book_id) % 4 != 0 else "Currently borrowed"


def render_book_card(row: pd.Series, key_prefix: str):
    title = row.get("title_clean", row.get("Title", "Untitled")) or "Untitled"
    author = row.get("author_clean", row.get("Author", "Unknown author")) or "Unknown author"
    subjects = row.get("subject_clean", row.get("Subjects", ""))
    isbn = first_isbn(row.get("ISBN Valid", ""))
    cover = openlibrary_cover_url(isbn, title, author)
    reason = row.get("reason", "Recommended for you.")
    availability = availability_label(row.get("i", 0))

    st.markdown('<div class="book-card">', unsafe_allow_html=True)
    if cover:
        st.image(cover, use_container_width=True)
    else:
        st.markdown('<div class="no-cover">No real cover found</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="book-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="book-author">{author}</div>', unsafe_allow_html=True)
    if subjects:
        st.markdown(f'<div class="book-meta">{subjects[:140]}{"…" if len(subjects) > 140 else ""}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="reason">Why: {reason}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="avail">{availability}</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("👍 Like", key=f"like_{key_prefix}_{row.get('i')}")
        st.button("💾 Save", key=f"save_{key_prefix}_{row.get('i')}")
    with c2:
        st.button("👎 Dislike", key=f"dislike_{key_prefix}_{row.get('i')}")
        st.button("Details", key=f"details_{key_prefix}_{row.get('i')}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_section(title: str, books: pd.DataFrame, key_prefix: str, max_books: int = 10):
    if books.empty:
        return
    st.subheader(title)
    books = books.head(max_books)
    cols = st.columns(5)
    for idx, (_, row) in enumerate(books.iterrows()):
        with cols[idx % 5]:
            render_book_card(row, f"{key_prefix}_{idx}")


def subject_options(items: pd.DataFrame) -> List[str]:
    values = []
    for text in items["Subjects"].dropna().head(3000):
        values.extend([x.strip() for x in str(text).split(";") if 3 < len(x.strip()) < 45])
    top = pd.Series(values).value_counts().head(40).index.tolist() if values else []
    return ["All subjects"] + top


def main():
    items = load_items()
    interactions = load_interactions()
    recs = load_recommendations()

    if "flow" not in st.session_state:
        st.session_state.flow = "Home"

    st.sidebar.title("📚 Library menu")
    st.sidebar.radio("User type", ["Home", "Existing user", "New user"], key="flow")
    st.sidebar.divider()
    search_query = st.sidebar.text_input("Search title, author, subject")
    subject_filter = st.sidebar.selectbox("Subject filter", subject_options(items))
    available_only = st.sidebar.checkbox("Available now only")

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Uni Library Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Discover books you might love, with clear reasons behind every recommendation.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.flow == "Home":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card"><b>Existing users</b><br><span class="small-muted">Use your library ID and borrowing history.</span></div>', unsafe_allow_html=True)
            if st.button("I already use the library", use_container_width=True):
                st.session_state.flow = "Existing user"
                st.rerun()
        with c2:
            st.markdown('<div class="metric-card"><b>New users</b><br><span class="small-muted">Answer a quick taste quiz.</span></div>', unsafe_allow_html=True)
            if st.button("I am new", use_container_width=True):
                st.session_state.flow = "New user"
                st.rerun()
        with c3:
            st.markdown(f'<div class="metric-card"><b>{len(items):,} books</b><br><span class="small-muted">{interactions["u"].nunique():,} users in history.</span></div>', unsafe_allow_html=True)
        st.info("Choose an entry flow from the buttons or sidebar.")
        return

    if st.session_state.flow == "Existing user":
        st.header("Existing user recommendations")
        user_id = st.number_input("Enter your library user ID", min_value=0, max_value=int(recs["user_id"].max()), value=1, step=1)
        if st.button("Get my recommendations", type="primary"):
            st.session_state.active_user_id = int(user_id)

        if "active_user_id" in st.session_state:
            uid = st.session_state.active_user_id
            history = history_for_user(uid, interactions, items, limit=10)
            seen = set(history["i"].dropna().astype(int).tolist()) if not history.empty else set()
            rec_row = recs[recs["user_id"] == uid]
            rec_ids = parse_recommendation_ids(rec_row.iloc[0]["recommendation"]) if not rec_row.empty else []
            personal = ids_to_books(rec_ids, items, "From your personalized recommendation model.", 12)

            st.success(f"Loaded profile for user {uid}.")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Books borrowed", int((interactions["u"] == uid).sum()))
            with m2:
                st.metric("Model recommendations", len(rec_ids))
            with m3:
                st.metric("Recent history shown", len(history))

            if not history.empty:
                st.subheader("Recent reading history")
                st.dataframe(history[["title_clean", "author_clean", "Subjects", "borrowed_at"]].rename(columns={"title_clean":"Title", "author_clean":"Author"}), use_container_width=True, hide_index=True)

            sections = [
                ("Recommended for you", personal, "personal"),
                ("Read again", ids_to_books(list(seen), items, "You borrowed this before — useful if you want to revisit it.", 10), "again"),
                ("Same authors", same_author_books(history, items, seen), "same_author"),
                ("Because of your subjects", similar_subject_books(history, items, seen), "subjects"),
                ("Popular now", popular_books(interactions, items, 12), "popular"),
            ]
            for title, books, key in sections:
                render_section(title, filter_books(books, search_query, subject_filter, available_only), key)

    if st.session_state.flow == "New user":
        st.header("New user onboarding")
        st.caption("Answer a few questions so we can create a starter taste profile.")
        with st.form("new_user_form"):
            language = st.selectbox("What language do you prefer to read?", ["Any", "French", "English", "German"])
            study_subject = st.text_input("What subject do you study?", placeholder="e.g. Psychology, Economics, Education")
            favorite_subjects = st.text_input("Favorite subject(s)", placeholder="Separate with commas")
            favorite_authors = st.text_input("Favorite author(s)", placeholder="Separate with commas")
            liked_books = st.text_input("Books you liked before", placeholder="Separate with commas")
            style = st.selectbox("Reading style", ["No preference", "Academic / study books", "Fiction / novels", "Short and easy reads"])
            submitted = st.form_submit_button("Generate recommendations", type="primary")

        if submitted:
            prefs = {
                "language": language,
                "subjects": [study_subject] + [x.strip() for x in favorite_subjects.split(",") if x.strip()],
                "authors": [x.strip() for x in favorite_authors.split(",") if x.strip()],
                "liked_books": [x.strip() for x in liked_books.split(",") if x.strip()],
                "style": style,
            }
            st.session_state.new_user_books = new_user_recommendations(items, interactions, prefs, 15)
            st.session_state.new_user_summary = prefs

        if "new_user_books" in st.session_state:
            summary = st.session_state.new_user_summary
            st.success("Starter profile created.")
            st.write("**Your taste profile:**", ", ".join([x for x in summary["subjects"] + summary["authors"] if x]) or "Popular library interests")
            new_books = filter_books(st.session_state.new_user_books, search_query, subject_filter, available_only)
            render_section("Recommended for you", new_books, "new_personal", 10)
            render_section("Popular now", filter_books(popular_books(interactions, items, 12), search_query, subject_filter, available_only), "new_popular")


if __name__ == "__main__":
    main()
