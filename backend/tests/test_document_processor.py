"""
Tests for DocumentProcessor.

Besides unit tests, golden snapshots of the real course documents pin down the
full parsing and chunking output, so any change to it shows up in review. After
an intentional output change, regenerate them with UPDATE_SNAPSHOTS=1.

Known bugs that are not fixed yet can be locked in by tests named
"pinned_quirk"; update those on purpose, in the same change as the fix.
"""

import json
import os
from pathlib import Path

import pytest

from document_processor import DocumentProcessor

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"


def write_doc(tmp_path: Path, text: str, name: str = "course.txt") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:

    def test_empty_and_whitespace_only_text(self):
        dp = DocumentProcessor(50, 10)
        assert dp.chunk_text("") == []
        assert dp.chunk_text("   \n\t  ") == []

    def test_whitespace_is_normalized(self):
        dp = DocumentProcessor(1000, 0)
        assert dp.chunk_text("  Hello\n\n  world.\tNext   one.  ") == ["Hello world. Next one."]

    def test_short_text_is_a_single_chunk(self):
        dp = DocumentProcessor(1000, 0)
        text = "First sentence. Second sentence. Third sentence."
        assert dp.chunk_text(text) == [text]

    def test_splits_only_at_sentence_boundaries(self):
        # chunk_size=1 forces one sentence per chunk, exposing the sentence splitter:
        # splits after . ! ? followed by a capital letter, but not after "Dr." or "e.g."
        dp = DocumentProcessor(1, 0)
        text = "First one. Second one! Third one? fourth lower. Dr. Smith came. See e.g. This."
        assert dp.chunk_text(text) == [
            "First one.",
            "Second one!",
            "Third one? fourth lower.",
            "Dr. Smith came.",
            "See e.g. This.",
        ]

    def test_sentence_longer_than_chunk_size_is_its_own_chunk(self):
        dp = DocumentProcessor(10, 0)
        assert dp.chunk_text("This sentence is way longer than ten. Short.") == [
            "This sentence is way longer than ten.",
            "Short.",
        ]

    def test_overlap_repeats_trailing_sentences(self):
        # "Four five." (10 chars) fits in the 15-char overlap, so it starts the next chunk.
        dp = DocumentProcessor(40, 15)
        chunks = dp.chunk_text("One two three. Four five. Six seven eight. Nine ten.")
        assert chunks[:2] == [
            "One two three. Four five.",
            "Four five. Six seven eight. Nine ten.",
        ]

    def test_no_redundant_tail_chunk(self):
        # Chunking stops at the chunk that reaches the end of the text; it does not
        # emit an extra chunk made only of overlap sentences it already contains.
        dp = DocumentProcessor(40, 15)
        chunks = dp.chunk_text("One two three. Four five. Six seven eight. Nine ten.")
        assert chunks == [
            "One two three. Four five.",
            "Four five. Six seven eight. Nine ten.",
        ]
        assert DocumentProcessor(30, 12).chunk_text("Aa bb. Cc dd. Ee ff gg hh. Ii.") == [
            "Aa bb. Cc dd. Ee ff gg hh. Ii.",
        ]
        # A text that fits in one chunk stays one chunk, even with overlap on
        assert DocumentProcessor(1000, 100).chunk_text("First sentence. Second sentence.") == [
            "First sentence. Second sentence.",
        ]
        assert DocumentProcessor(1000, 100).chunk_text("Only one sentence.") == ["Only one sentence."]

    def test_no_chunk_is_contained_in_the_previous_one(self):
        dp = DocumentProcessor(800, 100)
        text = (DOCS_DIR / "course1_script.txt").read_text(encoding="utf-8")
        chunks = dp.chunk_text(text)
        assert len(chunks) > 1
        assert not any(prev.endswith(chunk) for prev, chunk in zip(chunks, chunks[1:]))

    def test_no_overlap_chunks_rejoin_to_original(self):
        dp = DocumentProcessor(60, 0)
        text = " ".join(f"Sentence number {i} is here." for i in range(20))
        chunks = dp.chunk_text(text)
        assert len(chunks) > 1
        assert " ".join(chunks) == text

    def test_chunks_respect_size_unless_single_sentence(self):
        dp = DocumentProcessor(800, 100)
        text = (DOCS_DIR / "course1_script.txt").read_text(encoding="utf-8")
        splitter = DocumentProcessor(1, 0)  # one sentence per chunk
        for chunk in dp.chunk_text(text):
            assert len(chunk) <= 800 or len(splitter.chunk_text(chunk)) == 1


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_drops_invalid_utf8_bytes(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"Course Title: Caf\xe9\n")
    assert DocumentProcessor(800, 100).read_file(str(path)) == "Course Title: Caf\n"


# ---------------------------------------------------------------------------
# process_course_document
# ---------------------------------------------------------------------------


FULL_DOC = """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Ada Lovelace

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
Welcome to the course. This is the introduction.

Lesson 1: Basics
Lesson Link: https://example.com/lesson1
Here we learn the basics. They are important.

Lesson 2: Wrap-up
Final words here.
"""


class TestProcessCourseDocument:

    @pytest.fixture
    def parsed(self, tmp_path):
        # Same settings as config.py (CHUNK_SIZE=800, CHUNK_OVERLAP=100)
        return DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, FULL_DOC))

    def test_header_is_parsed(self, parsed):
        course, _ = parsed
        assert course.title == "Test Course"
        assert course.course_link == "https://example.com/course"
        assert course.instructor == "Ada Lovelace"

    def test_lessons_are_parsed(self, parsed):
        course, _ = parsed
        assert [(l.lesson_number, l.title, l.lesson_link) for l in course.lessons] == [
            (0, "Introduction", "https://example.com/lesson0"),
            (1, "Basics", "https://example.com/lesson1"),
            (2, "Wrap-up", None),
        ]

    def test_chunks_are_prefixed_and_numbered(self, parsed):
        _, chunks = parsed
        assert [c.content for c in chunks] == [
            "Course Test Course Lesson 0 content: Welcome to the course. This is the introduction.",
            "Course Test Course Lesson 1 content: Here we learn the basics. They are important.",
            "Course Test Course Lesson 2 content: Final words here.",
        ]
        assert [c.lesson_number for c in chunks] == [0, 1, 2]
        assert [c.chunk_index for c in chunks] == [0, 1, 2]
        assert all(c.course_title == "Test Course" for c in chunks)

    def test_lesson_link_line_is_not_content(self, parsed):
        _, chunks = parsed
        assert not any("Lesson Link" in c.content or "example.com" in c.content for c in chunks)

    def test_chunk_index_runs_across_lessons(self, tmp_path):
        body = " ".join(f"Sentence {i} of the lesson." for i in range(30))
        doc = f"Course Title: T\nCourse Link: L\nCourse Instructor: I\n\nLesson 1: A\n{body}\nLesson 2: B\n{body}\n"
        _, chunks = DocumentProcessor(100, 20).process_course_document(write_doc(tmp_path, doc))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert {c.lesson_number for c in chunks} == {1, 2}
        assert all(c.content.startswith(f"Course T Lesson {c.lesson_number} content: ") for c in chunks)

    def test_title_without_prefix_is_used_as_is(self, tmp_path):
        course, chunks = DocumentProcessor(800, 100).process_course_document(
            write_doc(tmp_path, "My Plain Title\n"))
        assert course.title == "My Plain Title"
        assert chunks == []

    def test_empty_file_falls_back_to_filename(self, tmp_path):
        course, chunks = DocumentProcessor(800, 100).process_course_document(
            write_doc(tmp_path, "", name="empty_course.txt"))
        assert course.title == "empty_course.txt"
        assert course.course_link is None
        assert course.instructor is None
        assert course.lessons == []
        assert chunks == []

    def test_missing_instructor_is_none(self, tmp_path):
        doc = "Course Title: T\nCourse Link: L\n\nLesson 1: A\nBody.\n"
        course, _ = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert course.course_link == "L"
        assert course.instructor is None

    def test_lesson_with_empty_body_is_dropped(self, tmp_path):
        doc = ("Course Title: T\nCourse Link: L\nCourse Instructor: I\n\n"
               "Lesson 1: Empty\n\nLesson 2: B\nLesson Link: http://b\nHello.\n")
        course, chunks = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert [l.lesson_number for l in course.lessons] == [2]
        assert [c.content for c in chunks] == ["Course T Lesson 2 content: Hello."]

    def test_no_lessons_fallback_is_prefixed(self, tmp_path):
        # Without "Lesson N:" markers, the whole body is chunked; chunks carry the
        # course title in their prefix but have no lesson number
        doc = "Course Title: T\nCourse Link: L\nCourse Instructor: I\n\nJust text. More text.\n"
        course, chunks = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert course.lessons == []
        assert [(c.content, c.lesson_number, c.chunk_index) for c in chunks] == [
            ("Course T content: Just text. More text.", None, 0),
        ]

    def test_body_starts_after_actual_header(self, tmp_path):
        # With a 2-line header, the "Lesson 1: A" marker on line 2 is part of the body
        doc = "Course Title: T\nCourse Instructor: I\nLesson 1: A\nBody one.\n"
        course, chunks = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert course.instructor == "I"
        assert [(l.lesson_number, l.title) for l in course.lessons] == [(1, "A")]
        assert [(c.content, c.lesson_number) for c in chunks] == [("Course T Lesson 1 content: Body one.", 1)]

    def test_title_only_header(self, tmp_path):
        doc = "Course Title: T\nLesson 1: A\nBody one.\n"
        course, chunks = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert (course.title, course.course_link, course.instructor) == ("T", None, None)
        assert [c.content for c in chunks] == ["Course T Lesson 1 content: Body one."]

    def test_header_fields_in_any_order_with_blank_lines(self, tmp_path):
        doc = "Course Title: T\n\nCourse Instructor: I\n\n\nCourse Link: L\nLesson 1: A\nBody one.\n"
        course, _ = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert (course.course_link, course.instructor) == ("L", "I")
        assert [l.lesson_number for l in course.lessons] == [1]

    def test_text_after_header_is_body(self, tmp_path):
        # A non-header line right after the title is body text: ignored before
        # the first lesson marker...
        doc = "Course Title: T\nA short description.\nLesson 1: A\nBody one.\n"
        course, chunks = DocumentProcessor(800, 100).process_course_document(write_doc(tmp_path, doc))
        assert [c.content for c in chunks] == ["Course T Lesson 1 content: Body one."]
        # ...but kept when the document has no lesson markers, even if it is
        # only two lines long
        course, chunks = DocumentProcessor(800, 100).process_course_document(
            write_doc(tmp_path, "My Title\nSome text here."))
        assert course.title == "My Title"
        assert [(c.content, c.lesson_number) for c in chunks] == [("Course My Title content: Some text here.", None)]


# ---------------------------------------------------------------------------
# Golden snapshots of the real course documents
# ---------------------------------------------------------------------------


def serialize(course, chunks) -> dict:
    return {
        "course": course.model_dump(),
        "chunks": [chunk.model_dump() for chunk in chunks],
    }


@pytest.mark.parametrize("doc_path", sorted(DOCS_DIR.glob("*.txt")), ids=lambda p: p.stem)
def test_real_docs_match_snapshot(doc_path):
    # Same settings as config.py (CHUNK_SIZE=800, CHUNK_OVERLAP=100).
    actual = serialize(*DocumentProcessor(800, 100).process_course_document(str(doc_path)))
    snapshot = SNAPSHOTS_DIR / f"{doc_path.stem}.json"

    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOTS_DIR.mkdir(exist_ok=True)
        snapshot.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    assert snapshot.exists(), f"Missing {snapshot.name}; run with UPDATE_SNAPSHOTS=1 to create it"
    assert actual == json.loads(snapshot.read_text(encoding="utf-8"))
