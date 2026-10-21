import os
import re
from typing import Iterator, List, Optional, Tuple
from models import Course, Lesson, CourseChunk

# Sentence boundary: whitespace after . ! or ? that is followed by a capital
# letter, except after abbreviations like "e.g." or "Dr."
SENTENCE_ENDINGS = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])')

# Document header and lesson markers (matched against stripped lines)
COURSE_TITLE = re.compile(r'^Course Title:\s*(.+)$', re.IGNORECASE)
COURSE_LINK = re.compile(r'^Course Link:\s*(.+)$', re.IGNORECASE)
COURSE_INSTRUCTOR = re.compile(r'^Course Instructor:\s*(.+)$', re.IGNORECASE)
LESSON_MARKER = re.compile(r'^Lesson\s+(\d+):\s*(.+)$', re.IGNORECASE)  # e.g. "Lesson 0: Introduction"
LESSON_LINK = re.compile(r'^Lesson Link:\s*(.+)$', re.IGNORECASE)

class DocumentProcessor:
    """Processes course documents and extracts structured information"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_file(self, file_path: str) -> str:
        """Read content from file with UTF-8 encoding"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into sentence-based chunks with overlap using config settings"""
        sentences = self._split_sentences(text)
        chunks = []
        start = 0
        while start < len(sentences):
            end = self._fill_chunk(sentences, start)
            chunk_sentences = sentences[start:end]
            chunks.append(' '.join(chunk_sentences))
            if end == len(sentences):
                break  # Reached the end; the overlap is already in this chunk
            # Next chunk starts with the overlap sentences, but always moves forward
            start = max(end - self._overlap_count(chunk_sentences), start + 1)
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Normalize whitespace and split text into non-empty sentences"""
        text = re.sub(r'\s+', ' ', text.strip())
        return [s.strip() for s in SENTENCE_ENDINGS.split(text) if s.strip()]

    def _fill_chunk(self, sentences: List[str], start: int) -> int:
        """
        Greedily add sentences, starting at sentences[start], while the joined
        chunk fits in chunk_size. The first sentence is always taken, even if
        it alone exceeds chunk_size.

        Returns:
            The index just past the last sentence in the chunk
        """
        size = len(sentences[start])
        end = start + 1
        while end < len(sentences) and size + 1 + len(sentences[end]) <= self.chunk_size:
            size += 1 + len(sentences[end])  # +1 for the joining space
            end += 1
        return end

    def _overlap_count(self, chunk_sentences: List[str]) -> int:
        """Number of trailing sentences of a chunk that fit in chunk_overlap"""
        overlap_size = 0
        count = 0
        for sentence in reversed(chunk_sentences):
            sentence_len = len(sentence) + (1 if count > 0 else 0)  # +1 for the joining space
            if overlap_size + sentence_len > self.chunk_overlap:
                break
            overlap_size += sentence_len
            count += 1
        return count

    def _make_chunks(self, text: str, course_title: str, lesson_number: Optional[int],
                     first_index: int, prefix: str = "") -> List[CourseChunk]:
        """
        Chunk text into CourseChunks, numbering them from first_index and
        prepending prefix to each chunk's content.
        """
        return [
            CourseChunk(
                content=prefix + chunk,
                course_title=course_title,
                lesson_number=lesson_number,
                chunk_index=first_index + offset
            )
            for offset, chunk in enumerate(self.chunk_text(text))
        ]

    def process_course_document(self, file_path: str) -> Tuple[Course, List[CourseChunk]]:
        """
        Process a course document with expected format:
        Line 1: Course Title: [title]
        Line 2: Course Link: [url]
        Line 3: Course Instructor: [instructor]
        Following lines: Lesson markers and content
        """
        content = self.read_file(file_path)
        lines = content.strip().split('\n')
        course, body_start = self._parse_header(lines, fallback_title=os.path.basename(file_path))

        course_chunks = []
        for lesson_number, lesson_title, lesson_link, lesson_content in self._split_lessons(lines, body_start):
            lesson_text = '\n'.join(lesson_content).strip()
            if not lesson_text:
                continue  # Lessons without content are dropped
            course.lessons.append(Lesson(
                lesson_number=lesson_number,
                title=lesson_title,
                lesson_link=lesson_link
            ))
            course_chunks.extend(self._make_chunks(
                lesson_text, course.title, lesson_number, first_index=len(course_chunks),
                prefix=f"Course {course.title} Lesson {lesson_number} content: "))

        # If no lessons found, treat entire content as one document
        if not course_chunks and len(lines) > 2:
            remaining_content = '\n'.join(lines[body_start:]).strip()
            course_chunks = self._make_chunks(remaining_content, course.title, None, first_index=0)

        return course, course_chunks

    @staticmethod
    def _parse_header(lines: List[str], fallback_title: str) -> Tuple[Course, int]:
        """
        Parse the course metadata at the top of the document.

        Returns:
            Tuple of (Course without lessons, index of the first body line)
        """
        # Line 0 holds the title, with or without a "Course Title:" label
        course_title = fallback_title
        first_line = lines[0].strip()
        if first_line:
            title_match = COURSE_TITLE.match(first_line)
            course_title = title_match.group(1).strip() if title_match else first_line

        # Lines 1-3 may hold the course link and the instructor, in any order
        course_link = None
        instructor_name = None
        for line in lines[1:4]:
            line = line.strip()
            if link_match := COURSE_LINK.match(line):
                course_link = link_match.group(1).strip()
            elif instructor_match := COURSE_INSTRUCTOR.match(line):
                instructor_name = instructor_match.group(1).strip()

        # Create course object with title as ID; an instructor given
        # literally as "Unknown" is treated as missing
        course = Course(
            title=course_title,
            course_link=course_link,
            instructor=None if instructor_name == "Unknown" else instructor_name
        )

        # The body is assumed to start at line 4 (after metadata), regardless
        # of how many header lines were actually found
        body_start = 3
        if len(lines) > 3 and not lines[3].strip():
            body_start = 4  # Skip empty line after instructor

        return course, body_start

    @staticmethod
    def _split_lessons(lines: List[str], start: int) -> Iterator[Tuple[int, str, Optional[str], List[str]]]:
        """
        Split the document body into lessons. Text before the first lesson
        marker is ignored.

        Yields:
            Tuples of (lesson number, lesson title, lesson link, content lines)
        """
        markers = [(i, match) for i in range(start, len(lines))
                   if (match := LESSON_MARKER.match(lines[i].strip()))]
        for k, (marker_index, lesson_match) in enumerate(markers):
            end = markers[k + 1][0] if k + 1 < len(markers) else len(lines)

            # An optional "Lesson Link:" line directly follows the marker
            content_start = marker_index + 1
            lesson_link = None
            if content_start < end:
                link_match = LESSON_LINK.match(lines[content_start].strip())
                if link_match:
                    lesson_link = link_match.group(1).strip()
                    content_start += 1

            yield (int(lesson_match.group(1)), lesson_match.group(2).strip(), lesson_link,
                   lines[content_start:end])
