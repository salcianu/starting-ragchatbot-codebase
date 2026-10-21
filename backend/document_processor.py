import os
import re
from typing import List, Optional, Tuple
from models import Course, Lesson, CourseChunk

# Sentence boundary: whitespace after . ! or ? that is followed by a capital
# letter, except after abbreviations like "e.g." or "Dr."
SENTENCE_ENDINGS = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])')

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

    def _add_lesson(self, course: Course, course_chunks: List[CourseChunk], chunk_counter: int,
                    lesson_number: Optional[int], lesson_title: Optional[str],
                    lesson_link: Optional[str], lesson_content: List[str]) -> int:
        """
        Add a lesson to the course and append its chunks to course_chunks.
        Every chunk is prefixed with the course title and lesson number.

        Returns:
            The updated chunk counter
        """
        if lesson_number is None or not lesson_content:
            return chunk_counter
        lesson_text = '\n'.join(lesson_content).strip()
        if not lesson_text:
            return chunk_counter

        course.lessons.append(Lesson(
            lesson_number=lesson_number,
            title=lesson_title,
            lesson_link=lesson_link
        ))

        for chunk in self.chunk_text(lesson_text):
            course_chunks.append(CourseChunk(
                content=f"Course {course.title} Lesson {lesson_number} content: {chunk}",
                course_title=course.title,
                lesson_number=lesson_number,
                chunk_index=chunk_counter
            ))
            chunk_counter += 1
        return chunk_counter

    def process_course_document(self, file_path: str) -> Tuple[Course, List[CourseChunk]]:
        """
        Process a course document with expected format:
        Line 1: Course Title: [title]
        Line 2: Course Link: [url]
        Line 3: Course Instructor: [instructor]
        Following lines: Lesson markers and content
        """
        content = self.read_file(file_path)
        filename = os.path.basename(file_path)
        
        lines = content.strip().split('\n')
        
        # Extract course metadata from first three lines
        course_title = filename  # Default fallback
        course_link = None
        instructor_name = "Unknown"
        
        # Parse course title from first line
        if len(lines) >= 1 and lines[0].strip():
            title_match = re.match(r'^Course Title:\s*(.+)$', lines[0].strip(), re.IGNORECASE)
            if title_match:
                course_title = title_match.group(1).strip()
            else:
                course_title = lines[0].strip()
        
        # Parse remaining lines for course metadata
        for i in range(1, min(len(lines), 4)):  # Check first 4 lines for metadata
            line = lines[i].strip()
            if not line:
                continue
                
            # Try to match course link
            link_match = re.match(r'^Course Link:\s*(.+)$', line, re.IGNORECASE)
            if link_match:
                course_link = link_match.group(1).strip()
                continue
                
            # Try to match instructor
            instructor_match = re.match(r'^Course Instructor:\s*(.+)$', line, re.IGNORECASE)
            if instructor_match:
                instructor_name = instructor_match.group(1).strip()
                continue
        
        # Create course object with title as ID
        course = Course(
            title=course_title,
            course_link=course_link,
            instructor=instructor_name if instructor_name != "Unknown" else None
        )
        
        # Process lessons and create chunks
        course_chunks = []
        current_lesson = None
        lesson_title = None
        lesson_link = None
        lesson_content = []
        chunk_counter = 0
        
        # Start processing from line 4 (after metadata)
        start_index = 3
        if len(lines) > 3 and not lines[3].strip():
            start_index = 4  # Skip empty line after instructor
        
        i = start_index
        while i < len(lines):
            line = lines[i]
            
            # Check for lesson markers (e.g., "Lesson 0: Introduction")
            lesson_match = re.match(r'^Lesson\s+(\d+):\s*(.+)$', line.strip(), re.IGNORECASE)
            
            if lesson_match:
                # Process previous lesson if it exists
                chunk_counter = self._add_lesson(course, course_chunks, chunk_counter,
                                                 current_lesson, lesson_title, lesson_link,
                                                 lesson_content)

                # Start new lesson
                current_lesson = int(lesson_match.group(1))
                lesson_title = lesson_match.group(2).strip()
                lesson_link = None
                
                # Check if next line is a lesson link
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    link_match = re.match(r'^Lesson Link:\s*(.+)$', next_line, re.IGNORECASE)
                    if link_match:
                        lesson_link = link_match.group(1).strip()
                        i += 1  # Skip the link line so it's not added to content
                
                lesson_content = []
            else:
                # Add line to current lesson content
                lesson_content.append(line)
                
            i += 1
        
        # Process the last lesson
        chunk_counter = self._add_lesson(course, course_chunks, chunk_counter,
                                         current_lesson, lesson_title, lesson_link,
                                         lesson_content)

        # If no lessons found, treat entire content as one document
        if not course_chunks and len(lines) > 2:
            remaining_content = '\n'.join(lines[start_index:]).strip()
            if remaining_content:
                chunks = self.chunk_text(remaining_content)
                for chunk in chunks:
                    course_chunk = CourseChunk(
                        content=chunk,
                        course_title=course.title,
                        chunk_index=chunk_counter
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1
        
        return course, course_chunks
