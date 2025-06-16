#!/usr/bin/env python3
"""
MCP-SFQM: Utility Functions Module
----------------------------------
유틸리티 모듈: 시스템 전체에서 사용되는 공통 함수 및 유틸리티 모음

이 모듈은 MCP-SFQM 시스템의 여러 구성 요소에서 공통적으로 사용되는 유틸리티 함수를 제공합니다.
파일 처리, 해싱, 비교, 정규식, 경로 조작 등의 기능을 포함합니다.

사용법:
    from mcp_sfqm.core.utils import (
        calculate_file_hash, compare_files, safe_parse_python, 
        find_files, sanitize_filename
    )
    
    # 파일 해시 계산
    file_hash = calculate_file_hash("path/to/file.py")
    
    # 파일 비교
    similarity = compare_files("file1.py", "file2.py")
    
    # 안전한 Python 파싱
    ast_tree, errors = safe_parse_python("path/to/file.py")
"""

import os
import sys
import re
import ast
import hashlib
import difflib
import tempfile
import shutil
import json
import yaml
import time
import importlib
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable, Iterator
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess

# 로거 설정
logger = logging.getLogger(__name__)

# 파일 시스템 관련 유틸리티
def find_files(
    base_dir: Union[str, Path], 
    pattern: str = "*.py", 
    recursive: bool = True,
    exclude_patterns: List[str] = None
) -> List[Path]:
    """지정된 패턴과 일치하는 파일 검색
    
    Args:
        base_dir: 검색 시작 디렉토리
        pattern: 검색할 파일 패턴 (glob 문법)
        recursive: 하위 디렉토리 검색 여부
        exclude_patterns: 제외할 패턴 목록
        
    Returns:
        일치하는 파일 경로 목록
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"디렉토리가 존재하지 않습니다: {base_dir}")
        return []
    
    # 제외 패턴 정규식 컴파일
    exclude_regex = None
    if exclude_patterns:
        exclude_strings = [f"({p})" for p in exclude_patterns]
        exclude_regex = re.compile("|".join(exclude_strings))
    
    # 파일 검색
    result = []
    if recursive:
        for path in base_path.rglob(pattern):
            # 제외 패턴 확인
            if exclude_regex and exclude_regex.search(str(path)):
                continue
            result.append(path)
    else:
        for path in base_path.glob(pattern):
            # 제외 패턴 확인
            if exclude_regex and exclude_regex.search(str(path)):
                continue
            result.append(path)
    
    return result


def ensure_directory(directory: Union[str, Path]) -> Path:
    """디렉토리 존재 확인 및 생성
    
    Args:
        directory: 생성할 디렉토리 경로
        
    Returns:
        생성된 디렉토리 경로
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_delete(path: Union[str, Path]) -> bool:
    """안전하게 파일 또는 디렉토리 삭제
    
    Args:
        path: 삭제할 경로
        
    Returns:
        삭제 성공 여부
    """
    try:
        target = Path(path)
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        return True
    except Exception as e:
        logger.error(f"삭제 실패: {path} - {e}")
        return False


def copy_with_metadata(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """메타데이터를 보존하며 파일 복사
    
    Args:
        src: 원본 파일 경로
        dst: 대상 파일 경로
        
    Returns:
        복사 성공 여부
    """
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"파일 복사 실패: {src} -> {dst} - {e}")
        return False


def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """파일 백업
    
    Args:
        file_path: 백업할 파일 경로
        backup_dir: 백업 디렉토리 (None이면 임시 디렉토리 사용)
        
    Returns:
        백업 파일 경로 또는 실패 시 None
    """
    try:
        source = Path(file_path)
        if not source.exists():
            logger.warning(f"백업할 파일이 존재하지 않습니다: {file_path}")
            return None
        
        # 백업 디렉토리 설정
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
        else:
            backup_path = Path(tempfile.gettempdir()) / "mcp_sfqm_backups"
            backup_path.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프 추가 백업 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"{source.stem}_{timestamp}{source.suffix}"
        
        # 복사
        shutil.copy2(source, backup_file)
        logger.debug(f"파일 백업 완료: {source} -> {backup_file}")
        
        return backup_file
    
    except Exception as e:
        logger.error(f"파일 백업 실패: {file_path} - {e}")
        return None


def sanitize_filename(filename: str) -> str:
    """파일명에서 유효하지 않은 문자 제거
    
    Args:
        filename: 정리할 파일명
        
    Returns:
        정리된 파일명
    """
    # 파일 시스템에서 유효하지 않은 문자 제거
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # 최대 길이 제한
    max_length = 240  # 일반적인 파일 시스템 제한보다 적게
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    return sanitized


# 해시 및 비교 관련 유틸리티
def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """파일의 해시값 계산
    
    Args:
        file_path: 해시를 계산할 파일 경로
        algorithm: 해시 알고리즘 (md5, sha1, sha256, sha512)
        chunk_size: 읽기 버퍼 크기
        
    Returns:
        계산된 해시값 (16진수 문자열)
    """
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"파일 해시 계산 실패: {file_path} - {e}")
        return ""


def calculate_string_hash(text: str, algorithm: str = "sha256") -> str:
    """문자열의 해시값 계산
    
    Args:
        text: 해시를 계산할 문자열
        algorithm: 해시 알고리즘 (md5, sha1, sha256, sha512)
        
    Returns:
        계산된 해시값 (16진수 문자열)
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def compare_files(file1: Union[str, Path], file2: Union[str, Path]) -> float:
    """두 파일의 유사도 계산
    
    Args:
        file1: 첫 번째 파일 경로
        file2: 두 번째 파일 경로
        
    Returns:
        유사도 (0.0 ~ 1.0)
    """
    try:
        with open(file1, 'r', encoding='utf-8', errors='ignore') as f1, \
             open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
            text1 = f1.read()
            text2 = f2.read()
            
        # difflib의 SequenceMatcher로 유사도 계산
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    except Exception as e:
        logger.error(f"파일 비교 실패: {file1}, {file2} - {e}")
        return 0.0


def calculate_text_similarity(text1: str, text2: str, method: str = "sequence") -> float:
    """두 텍스트의 유사도 계산
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        method: 유사도 계산 방법 (sequence, jaccard, cosine)
        
    Returns:
        유사도 (0.0 ~ 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    if method == "sequence":
        # 순서 고려한 유사도 (difflib)
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    elif method == "jaccard":
        # 자카드 유사도 (집합 기반)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    elif method == "cosine":
        # 코사인 유사도 (벡터 기반)
        # 간단한 구현 (실제 프로젝트에서는 더 효율적인 구현 권장)
        words = set(text1.lower().split()).union(set(text2.lower().split()))
        words_dict = {word: i for i, word in enumerate(words)}
        
        # 각 텍스트를 벡터로 변환
        vec1 = [0] * len(words_dict)
        vec2 = [0] * len(words_dict)
        
        for word in text1.lower().split():
            vec1[words_dict[word]] += 1
        
        for word in text2.lower().split():
            vec2[words_dict[word]] += 1
        
        # 코사인 유사도 계산
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = sum(v ** 2 for v in vec1) ** 0.5
        magnitude2 = sum(v ** 2 for v in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    else:
        logger.warning(f"지원되지 않는 유사도 계산 방법: {method}")
        return 0.0


# Python 코드 처리 유틸리티
def safe_parse_python(file_path: Union[str, Path]) -> Tuple[Optional[ast.AST], List[str]]:
    """안전하게 Python 파일 파싱
    
    Args:
        file_path: 파싱할 파일 경로
        
    Returns:
        (AST 트리, 오류 메시지 목록) 튜플
    """
    errors = []
    ast_tree = None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        try:
            # Python 3 문법으로 파싱 시도
            ast_tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            errors.append(f"구문 오류: {e}")
            
            # 일반적인 구문 오류 수정 시도
            fixed_source = fix_common_syntax_errors(source)
            
            if fixed_source != source:
                try:
                    # 수정된 코드로 다시 파싱
                    ast_tree = ast.parse(fixed_source, filename=str(file_path))
                    errors.append("자동 수정 후 파싱 성공")
                except SyntaxError as e2:
                    errors.append(f"자동 수정 후에도 구문 오류 발생: {e2}")
    
    except Exception as e:
        errors.append(f"파일 읽기 또는 파싱 중 오류: {e}")
    
    return ast_tree, errors


def fix_common_syntax_errors(source: str) -> str:
    """일반적인 Python 구문 오류 자동 수정
    
    Args:
        source: 원본 소스 코드
        
    Returns:
        수정된 소스 코드
    """
    fixed = source
    
    # 누락된 괄호 닫기 처리
    brackets = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    
    # 각 괄호 유형별 개수 확인
    for opener, closer in brackets.items():
        open_count = fixed.count(opener)
        close_count = fixed.count(closer)
        
        # 닫는 괄호가 부족한 경우 추가
        if open_count > close_count:
            fixed += closer * (open_count - close_count)
    
    # 누락된 콜론 처리
    # if, else, elif, for, while, def, class 문 뒤에 콜론이 없는 경우
    colon_patterns = [
        (r'(if\s+[^:]+)\s*\n', r'\1:\n'),
        (r'(elif\s+[^:]+)\s*\n', r'\1:\n'),
        (r'(else)\s*\n', r'\1:\n'),
        (r'(for\s+[^:]+)\s*\n', r'\1:\n'),
        (r'(while\s+[^:]+)\s*\n', r'\1:\n'),
        (r'(def\s+\w+\([^)]*\))\s*\n', r'\1:\n'),
        (r'(class\s+\w+(?:\([^)]*\))?)\s*\n', r'\1:\n')
    ]
    
    for pattern, replacement in colon_patterns:
        fixed = re.sub(pattern, replacement, fixed)
    
    # 들여쓰기 일관성 처리
    lines = fixed.split('\n')
    indentation_fixed_lines = []
    
    for i, line in enumerate(lines):
        if i > 0 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            # 이전 줄이 콜론으로 끝나면 들여쓰기 추가
            prev_line = lines[i-1].strip()
            if prev_line.endswith(':'):
                indentation_fixed_lines.append('    ' + line)
                continue
        
        indentation_fixed_lines.append(line)
    
    fixed = '\n'.join(indentation_fixed_lines)
    
    return fixed


def get_functions_from_ast(ast_tree: ast.AST) -> List[ast.FunctionDef]:
    """AST에서 함수 정의 추출
    
    Args:
        ast_tree: AST 트리
        
    Returns:
        함수 정의 노드 목록
    """
    functions = []
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
    
    return functions


def get_classes_from_ast(ast_tree: ast.AST) -> List[ast.ClassDef]:
    """AST에서 클래스 정의 추출
    
    Args:
        ast_tree: AST 트리
        
    Returns:
        클래스 정의 노드 목록
    """
    classes = []
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node)
    
    return classes


def get_imports_from_ast(ast_tree: ast.AST) -> List[Tuple[str, Optional[str]]]:
    """AST에서 임포트 문 추출
    
    Args:
        ast_tree: AST 트리
        
    Returns:
        (모듈명, 별칭) 튜플 목록
    """
    imports = []
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append((name.name, name.asname))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for name in node.names:
                if module:
                    full_name = f"{module}.{name.name}"
                else:
                    full_name = name.name
                imports.append((full_name, name.asname))
    
    return imports


def extract_docstring(node: Union[ast.FunctionDef, ast.ClassDef, ast.Module]) -> Optional[str]:
    """AST 노드에서 문서 문자열 추출
    
    Args:
        node: AST 노드 (함수, 클래스, 모듈)
        
    Returns:
        문서 문자열 또는 None
    """
    if not node.body:
        return None
    
    first_node = node.body[0]
    if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Str):
        return first_node.value.s
    
    return None


def get_complexity_metrics(ast_tree: ast.AST) -> Dict[str, Any]:
    """코드 복잡도 메트릭 계산
    
    Args:
        ast_tree: AST 트리
        
    Returns:
        복잡도 메트릭 딕셔너리
    """
    # 기본 메트릭 초기화
    metrics = {
        "lines_of_code": 0,
        "cyclomatic_complexity": 0,
        "max_nesting_depth": 0,
        "num_functions": 0,
        "num_classes": 0,
        "num_methods": 0,
        "avg_function_length": 0,
        "max_function_length": 0,
        "function_complexities": {}
    }
    
    # 함수 및 클래스 수집
    functions = []
    classes = []
    methods = []
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.FunctionDef):
            # 클래스 메서드 구분
            parent_class = False
            for parent in ast.walk(ast_tree):
                if isinstance(parent, ast.ClassDef) and node in parent.body:
                    parent_class = True
                    methods.append(node)
                    break
            
            if not parent_class:
                functions.append(node)
        
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
    
    # 메트릭 계산
    metrics["num_functions"] = len(functions)
    metrics["num_classes"] = len(classes)
    metrics["num_methods"] = len(methods)
    
    # 순환 복잡도 계산
    cc_visitor = CyclomaticComplexityVisitor()
    cc_visitor.visit(ast_tree)
    metrics["cyclomatic_complexity"] = cc_visitor.complexity
    
    # 함수별 복잡도
    function_lengths = []
    for func in functions + methods:
        func_name = func.name
        func_complexity = calculate_function_complexity(func)
        metrics["function_complexities"][func_name] = func_complexity
        
        # 함수 길이 계산
        func_lines = len(ast.unparse(func).split("\n"))
        function_lengths.append(func_lines)
    
    # 평균 및 최대 함수 길이
    if function_lengths:
        metrics["avg_function_length"] = sum(function_lengths) / len(function_lengths)
        metrics["max_function_length"] = max(function_lengths)
    
    # 중첩 깊이 계산
    nesting_visitor = NestingDepthVisitor()
    nesting_visitor.visit(ast_tree)
    metrics["max_nesting_depth"] = nesting_visitor.max_depth
    
    # 코드 라인 수 계산
    metrics["lines_of_code"] = len(ast.unparse(ast_tree).split("\n"))
    
    return metrics


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """순환 복잡도 계산 방문자"""
    
    def __init__(self):
        self.complexity = 1  # 기본값은 1
    
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        # 'and' 및 'or' 연산자는 복잡도에 영향을 줄 수 있음
        if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):
            self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        self.complexity += len(node.handlers)
        self.generic_visit(node)


class NestingDepthVisitor(ast.NodeVisitor):
    """중첩 깊이 계산 방문자"""
    
    def __init__(self):
        self.current_depth = 0
        self.max_depth = 0
    
    def visit_FunctionDef(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_ClassDef(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_If(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_For(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_While(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_Try(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1


def calculate_function_complexity(func_node: ast.FunctionDef) -> Dict[str, Any]:
    """함수의 복잡도 메트릭 계산
    
    Args:
        func_node: 함수 정의 AST 노드
        
    Returns:
        복잡도 메트릭 딕셔너리
    """
    # 순환 복잡도
    cc_visitor = CyclomaticComplexityVisitor()
    cc_visitor.visit(func_node)
    cyclomatic_complexity = cc_visitor.complexity
    
    # 중첩 깊이
    nesting_visitor = NestingDepthVisitor()
    nesting_visitor.visit(func_node)
    max_nesting = nesting_visitor.max_depth
    
    # 파라미터 수
    num_params = len(func_node.args.args)
    
    # 리턴문 수
    return_count = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            return_count += 1
    
    # 함수 길이 (라인 수)
    func_lines = len(ast.unparse(func_node).split("\n"))
    
    # 문서화 여부
    has_docstring = extract_docstring(func_node) is not None
    
    return {
        "cyclomatic_complexity": cyclomatic_complexity,
        "max_nesting": max_nesting,
        "params_count": num_params,
        "return_count": return_count,
        "lines": func_lines,
        "has_docstring": has_docstring
    }


# 포맷 및 직렬화 유틸리티
def to_json(obj: Any, pretty: bool = True) -> str:
    """객체를 JSON 문자열로 변환
    
    Args:
        obj: 변환할 객체
        pretty: 들여쓰기 적용 여부
        
    Returns:
        JSON 문자열
    """
    if pretty:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_serializer)
    return json.dumps(obj, ensure_ascii=False, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """JSON 직렬화 불가능한 객체 처리
    
    Args:
        obj: 직렬화할 객체
        
    Returns:
        직렬화 가능한 표현
    """
    # 경로 객체 처리
    if isinstance(obj, Path):
        return str(obj)
    
    # 날짜/시간 객체 처리
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # 집합 처리
    if isinstance(obj, set):
        return list(obj)
    
    # 그 외 직렬화 불가능한 객체
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    
    # 기본 처리
    return str(obj)


def load_json(file_path: Union[str, Path]) -> Any:
    """JSON 파일 로드
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        로드된 객체
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON 파일 로드 실패: {file_path} - {e}")
        return None


def load_yaml(file_path: Union[str, Path]) -> Any:
    """YAML 파일 로드
    
    Args:
        file_path: YAML 파일 경로
        
    Returns:
        로드된 객체
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"YAML 파일 로드 실패: {file_path} - {e}")
        return None


# 병렬 처리 유틸리티
def parallel_map(func: Callable, items: List[Any], max_workers: int = None, use_processes: bool = False) -> List[Any]:
    """함수를 항목 목록에 병렬로 적용
    
    Args:
        func: 적용할 함수
        items: 입력 항목 목록
        max_workers: 최대 작업자 수 (None이면 CPU 코어 수 사용)
        use_processes: 프로세스 사용 여부 (True면 프로세스, False면 스레드)
        
    Returns:
        결과 목록
    """
    if not items:
        return []
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results


# 시스템 정보 유틸리티
def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집
    
    Returns:
        시스템 정보 딕셔너리
    """
    import platform
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "user": os.getlogin() if hasattr(os, 'getlogin') else 'unknown',
        "timestamp": datetime.now().isoformat()
    }
    
    # 메모리 정보 (psutil 패키지가 있는 경우)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory"] = mem.total
        info["available_memory"] = mem.available
    except ImportError:
        pass
    
    return info


# 모듈 동적 로딩 유틸리티
def load_module_from_path(module_path: Union[str, Path], module_name: Optional[str] = None) -> Any:
    """파일 경로에서 Python 모듈 동적 로드
    
    Args:
        module_path: 모듈 파일 경로
        module_name: 모듈 이름 (None이면 파일명에서 추출)
        
    Returns:
        로드된 모듈 또는 실패 시 None
    """
    try:
        path = Path(module_path)
        
        if not module_name:
            module_name = path.stem
        
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec:
            logger.error(f"모듈 스펙 생성 실패: {module_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
    
    except Exception as e:
        logger.error(f"모듈 로드 실패: {module_path} - {e}")
        return None


# 명령 실행 유틸리티
def run_command(command: Union[str, List[str]], shell: bool = False, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """외부 명령 실행
    
    Args:
        command: 실행할 명령 (문자열 또는 리스트)
        shell: 쉘 사용 여부
        timeout: 타임아웃 (초)
        
    Returns:
        (반환 코드, 표준 출력, 표준 오류) 튜플
    """
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"명령 실행 시간 초과: {command}")
        return -1, "", "Timeout expired"
    
    except Exception as e:
        logger.error(f"명령 실행 실패: {command} - {e}")
        return -1, "", str(e)


if __name__ == "__main__":
    # 모듈 테스트 코드
    logging.basicConfig(level=logging.DEBUG)
    
    # 파일 검색 테스트
    print("현재 디렉토리의 Python 파일:")
    for file in find_files(".", "*.py"):
        print(f"  {file}")
    
    # 텍스트 유사도 테스트
    text1 = "This is a test sentence for similarity comparison."
    text2 = "This is a test phrase for similarity measurement."
    
    print("\n텍스트 유사도 테스트:")
    print(f"시퀀스 유사도: {calculate_text_similarity(text1, text2, 'sequence'):.4f}")
    print(f"자카드 유사도: {calculate_text_similarity(text1, text2, 'jaccard'):.4f}")
    print(f"코사인 유사도: {calculate_text_similarity(text1, text2, 'cosine'):.4f}")
    
    # 시스템 정보 테스트
    print("\n시스템 정보:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
