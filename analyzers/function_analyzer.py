#!/usr/bin/env python3
"""
MCP-SFQM: Function Analyzer Module
----------------------------------
함수 분석기 모듈: MCP 서버 함수 단위 분석 수행

이 모듈은 MCP 서버 파일에서 함수를 추출하고 세부적인 함수 단위 분석을 수행합니다.
함수의 구조, 복잡도, 매개변수, 반환 값, 에러 처리 등 다양한 측면을 분석합니다.

특징:
- 함수 식별 및 메타데이터 추출
- 함수 복잡도 분석 (순환, 인지적, 매개변수)
- 함수 품질 평가 (문서화, 에러 처리, 유형 힌트)
- MCP 프로토콜 호환성 확인
- 함수 간 의존성 분석

사용법:
    from mcp_sfqm.analyzers.function_analyzer import FunctionAnalyzer
    
    analyzer = FunctionAnalyzer()
    result = analyzer.run(base_dir="path/to/mcp_servers")
    
    # 함수 분석 결과 접근
    for func_info in result.results.get("functions", []):
        print(f"함수: {func_info['name']}, 복잡도: {func_info['complexity']}")
"""

import os
import ast
import re
import logging
import inspect
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from pathlib import Path
import time
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# 상위 패키지 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_sfqm.analyzers.base_analyzer import BaseAnalyzer, AnalysisContext
from mcp_sfqm.core.logging_manager import LoggingManager
from mcp_sfqm.core.utils import (
    find_files, safe_parse_python, get_functions_from_ast, 
    extract_docstring, calculate_function_complexity
)
from mcp_sfqm.core.exceptions import FunctionAnalysisError

# 로거 설정
logger = LoggingManager.get_logger(__name__, with_context=True)


@dataclass
class MCPFunctionInfo:
    """MCP 함수 정보"""
    
    # 기본 정보
    name: str
    file_path: Path
    line_number: int
    end_line_number: int
    source_code: str
    ast_node: Optional[ast.FunctionDef] = None
    
    # 문서 및 메타데이터
    docstring: Optional[str] = None
    description: str = ""
    is_public: bool = True
    is_method: bool = False
    class_name: Optional[str] = None
    
    # 매개변수 정보
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type_hint: Optional[str] = None
    has_self_param: bool = False
    has_kwargs: bool = False
    has_args: bool = False
    default_param_count: int = 0
    
    # 복잡도 지표
    complexity: Dict[str, Any] = field(default_factory=dict)
    
    # 품질 지표
    has_docstring: bool = False
    has_type_hints: bool = False
    has_return_annotation: bool = False
    has_error_handling: bool = False
    error_handling_score: float = 0.0
    validation_score: float = 0.0
    
    # MCP 관련 지표
    is_mcp_handler: bool = False
    is_mcp_core_function: bool = False
    mcp_protocol_level: str = "unknown"  # unknown, basic, standard, advanced
    
    # 의존성
    calls: List[str] = field(default_factory=list)
    imported_modules: List[str] = field(default_factory=list)
    
    # 테스트 관련
    has_tests: bool = False
    test_files: List[Path] = field(default_factory=list)
    
    def __post_init__(self):
        """초기화 후처리"""
        # 소스 코드에서 문서 문자열 추출
        if self.ast_node and not self.docstring:
            self.docstring = extract_docstring(self.ast_node)
            self.has_docstring = bool(self.docstring)
            
            # 문서 문자열에서 설명 추출
            if self.docstring:
                self.description = self._extract_description_from_docstring()
        
        # 매개변수 정보 추출
        if self.ast_node and not self.parameters:
            self._extract_parameter_info()
        
        # 공개 여부 확인
        if self.name.startswith('_'):
            self.is_public = False
    
    def _extract_description_from_docstring(self) -> str:
        """문서 문자열에서 첫 줄 설명 추출"""
        if not self.docstring:
            return ""
            
        # 첫 줄만 추출
        first_line = self.docstring.strip().split('\n')[0].strip()
        
        # 큰따옴표 제거
        first_line = first_line.strip('"\'')
        
        return first_line
    
    def _extract_parameter_info(self) -> None:
        """AST에서 매개변수 정보 추출"""
        if not self.ast_node:
            return
            
        args = self.ast_node.args
        
        # self 매개변수 확인
        if args.args and args.args[0].arg == 'self':
            self.has_self_param = True
            self.is_method = True
        
        # 가변 인자 확인
        self.has_args = args.vararg is not None
        self.has_kwargs = args.kwarg is not None
        
        # 기본값 개수 확인
        self.default_param_count = len(args.defaults)
        
        # 매개변수 목록 작성
        param_list = []
        
        # 일반 매개변수
        for i, arg in enumerate(args.args):
            param = {
                "name": arg.arg,
                "position": i,
                "has_default": i >= len(args.args) - self.default_param_count,
                "has_type_hint": arg.annotation is not None
            }
            
            # 타입 힌트 확인
            if arg.annotation:
                param["type_hint"] = ast.unparse(arg.annotation)
                self.has_type_hints = True
            
            # 기본값 확인
            if param["has_default"]:
                default_idx = i - (len(args.args) - self.default_param_count)
                param["default_value"] = ast.unparse(args.defaults[default_idx])
            
            param_list.append(param)
        
        # 가변 위치 인자
        if args.vararg:
            param_list.append({
                "name": f"*{args.vararg.arg}",
                "position": len(param_list),
                "has_default": False,
                "has_type_hint": args.vararg.annotation is not None,
                "is_vararg": True
            })
            
            if args.vararg.annotation:
                param_list[-1]["type_hint"] = ast.unparse(args.vararg.annotation)
                self.has_type_hints = True
        
        # 키워드 전용 인자
        for i, arg in enumerate(args.kwonlyargs):
            param = {
                "name": arg.arg,
                "position": len(param_list),
                "has_default": i < len(args.kw_defaults) and args.kw_defaults[i] is not None,
                "has_type_hint": arg.annotation is not None,
                "is_kwonly": True
            }
            
            # 타입 힌트 확인
            if arg.annotation:
                param["type_hint"] = ast.unparse(arg.annotation)
                self.has_type_hints = True
            
            # 기본값 확인
            if param["has_default"] and args.kw_defaults[i] is not None:
                param["default_value"] = ast.unparse(args.kw_defaults[i])
            
            param_list.append(param)
        
        # 가변 키워드 인자
        if args.kwarg:
            param_list.append({
                "name": f"**{args.kwarg.arg}",
                "position": len(param_list),
                "has_default": False,
                "has_type_hint": args.kwarg.annotation is not None,
                "is_kwarg": True
            })
            
            if args.kwarg.annotation:
                param_list[-1]["type_hint"] = ast.unparse(args.kwarg.annotation)
                self.has_type_hints = True
        
        self.parameters = param_list
        
        # 반환 타입 힌트 확인
        if self.ast_node.returns:
            self.return_type_hint = ast.unparse(self.ast_node.returns)
            self.has_return_annotation = True
            self.has_type_hints = True
    
    def analyze_error_handling(self) -> None:
        """함수의 에러 처리 분석"""
        if not self.ast_node:
            return
        
        try_count = 0
        except_count = 0
        finally_count = 0
        raise_count = 0
        
        # 노드 순회하며 에러 처리 관련 요소 카운트
        for node in ast.walk(self.ast_node):
            if isinstance(node, ast.Try):
                try_count += 1
                except_count += len(node.handlers)
                finally_count += 1 if node.finalbody else 0
            elif isinstance(node, ast.Raise):
                raise_count += 1
        
        # 에러 처리 여부 및 점수 계산
        self.has_error_handling = try_count > 0 or raise_count > 0
        
        # 에러 처리 점수 계산 (0.0 ~ 1.0)
        if self.has_error_handling:
            # 간단한 공식으로 점수 계산
            self.error_handling_score = min(1.0, (try_count * 0.3 + except_count * 0.3 + 
                                             finally_count * 0.2 + raise_count * 0.2))
        else:
            self.error_handling_score = 0.0
    
    def analyze_validation(self) -> None:
        """함수의 입력 검증 분석"""
        if not self.ast_node or not self.source_code:
            return
            
        validation_patterns = [
            # 타입 검사
            r'isinstance\s*\(',
            r'type\s*\(',
            # None 검사
            r'is\s+None',
            r'is\s+not\s+None',
            # 값 검사
            r'if\s+.*\s*[<>=!]=',
            r'assert\s+',
            # 길이 검사
            r'len\s*\(',
        ]
        
        validation_count = 0
        
        # 패턴 검색
        for pattern in validation_patterns:
            matches = re.findall(pattern, self.source_code)
            validation_count += len(matches)
        
        # 매개변수 수 기준 검증
        param_count = len(self.parameters)
        if self.has_self_param and param_count > 0:
            param_count -= 1  # self 제외
            
        # 점수 계산 (0.0 ~ 1.0)
        if param_count > 0:
            # 매개변수 당 검증 수와 총 검증 수를 고려한 점수
            self.validation_score = min(1.0, (validation_count / param_count) * 0.7 + 
                                       (0.3 if validation_count > 0 else 0.0))
        else:
            # 매개변수가 없으면 검증 필요성이 낮음
            self.validation_score = 1.0
    
    def check_mcp_handler(self) -> None:
        """MCP 핸들러 함수인지 확인"""
        if not self.name or not self.docstring:
            return
        
        # MCP 핸들러 함수 이름 패턴
        mcp_handler_patterns = [
            r'handle_',
            r'process_',
            r'execute_',
            r'run_model'
        ]
        
        # 이름 검사
        for pattern in mcp_handler_patterns:
            if re.match(pattern, self.name):
                self.is_mcp_handler = True
                break
        
        # 문서 문자열 검사
        mcp_doc_patterns = [
            r'MCP',
            r'Model Context Protocol',
            r'model\s+request',
            r'model\s+response'
        ]
        
        for pattern in mcp_doc_patterns:
            if re.search(pattern, self.docstring, re.IGNORECASE):
                self.is_mcp_handler = True
                break
        
        # 매개변수 검사
        param_names = [p['name'] for p in self.parameters]
        mcp_param_indicators = ['model_name', 'request', 'context', 'prompt', 'input_data']
        
        if any(p in param_names for p in mcp_param_indicators):
            self.is_mcp_handler = True
    
    def check_mcp_core_function(self) -> None:
        """MCP 핵심 함수인지 확인"""
        if not self.name:
            return
        
        # MCP 핵심 함수 목록
        mcp_core_functions = [
            'get_model_names',
            'get_model_metadata',
            'execute',
            'generate',
            'process_request',
            'handle_mcp_request',
            'run_model',
            'get_available_models'
        ]
        
        if self.name in mcp_core_functions:
            self.is_mcp_core_function = True
            
            # 프로토콜 레벨 설정
            if self.name in ['get_model_names', 'get_model_metadata', 'execute']:
                self.mcp_protocol_level = "basic"
            elif self.name in ['generate', 'process_request']:
                self.mcp_protocol_level = "standard"
            else:
                self.mcp_protocol_level = "advanced"
    
    def extract_function_calls(self) -> None:
        """함수 내부에서 호출하는 다른 함수 추출"""
        if not self.ast_node:
            return
            
        calls = []
        
        # 노드 순회하며 함수 호출 검색
        for node in ast.walk(self.ast_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # 직접 함수 호출 (func_name())
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # 메서드 호출 (obj.method())
                    if isinstance(node.func.value, ast.Name):
                        calls.append(f"{node.func.value.id}.{node.func.attr}")
        
        self.calls = calls
    
    def to_dict(self) -> Dict[str, Any]:
        """데이터를 딕셔너리로 변환"""
        data = asdict(self)
        
        # AST 노드는 직렬화 불가능하므로 제외
        data.pop('ast_node', None)
        
        # 경로를 문자열로 변환
        data['file_path'] = str(data['file_path'])
        data['test_files'] = [str(p) for p in data['test_files']]
        
        return data


class FunctionAnalyzer(BaseAnalyzer):
    """함수 분석기
    
    MCP 서버 파일의 함수를 추출하고 분석합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """함수 분석기 초기화
        
        Args:
            config: 분석기 설정
        """
        super().__init__(config=config)
        
        # 기본 설정
        self.default_config = {
            "file_patterns": ["*.py"],
            "exclude_patterns": ["*test*", "*__pycache__*"],
            "min_function_size": 3,  # 최소 3줄 이상
            "analyze_error_handling": True,
            "analyze_validation": True,
            "check_mcp_compatibility": True,
            "extract_calls": True,
            "find_tests": True
        }
        
        # 설정 병합
        if self.config:
            self.default_config.update(self.config)
        self.config = self.default_config
    
    def analyze(self, context: AnalysisContext) -> AnalysisContext:
        """함수 분석 실행
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            업데이트된 분석 컨텍스트
        """
        self.logger.info("함수 분석 시작", context={"analyzer": self.__class__.__name__})
        
        # 기본 디렉토리 확인
        if not context.base_dir:
            error_msg = "분석할 기본 디렉토리가 지정되지 않았습니다"
            self.add_error(context, error_msg)
            return context
        
        # 파일 목록 가져오기
        target_files = self._get_target_files(context)
        if not target_files:
            warning_msg = "분석할 파일을 찾을 수 없습니다"
            self.add_warning(context, warning_msg)
            return context
            
        self.logger.info(f"분석 대상 파일: {len(target_files)}개", 
                        context={"analyzer": self.__class__.__name__})
        
        # 대상 파일 설정
        context.target_files = target_files
        
        # 진행 상황 업데이트
        self.update_progress(context, 0.1, "파일 목록 수집 완료")
        
        # 병렬 분석
        file_count = len(target_files)
        all_functions = []
        
        for i, file_path in enumerate(target_files):
            # 진행 상황 업데이트
            progress = 0.1 + (0.8 * (i + 1) / file_count)
            self.update_progress(context, progress, f"파일 분석 중: {i+1}/{file_count}")
            
            try:
                functions = self._analyze_file(file_path)
                all_functions.extend(functions)
            except Exception as e:
                error_msg = f"파일 분석 중 오류 발생: {file_path}"
                self.add_error(context, error_msg, {"exception": str(e), "file": str(file_path)})
        
        # 결과 추가
        self.add_result(context, "functions", [f.to_dict() for f in all_functions])
        self.add_result(context, "function_count", len(all_functions))
        
        # 요약 정보 추가
        summary = self._generate_summary(all_functions)
        self.add_result(context, "summary", summary)
        
        # 주요 지표 추가
        metrics = self._calculate_metrics(all_functions)
        for key, value in metrics.items():
            context.add_metric(key, value)
        
        # 진행 상황 업데이트
        self.update_progress(context, 1.0, "함수 분석 완료")
        
        self.logger.info(f"함수 분석 완료: {len(all_functions)}개 함수 분석됨", 
                        context={"analyzer": self.__class__.__name__})
        
        return context
    
    def _get_target_files(self, context: AnalysisContext) -> List[Path]:
        """분석 대상 파일 목록 가져오기
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            대상 파일 경로 목록
        """
        # 이미 대상 파일이 지정된 경우
        if context.target_files:
            return context.target_files
            
        # 파일 패턴으로 검색
        patterns = self.config.get("file_patterns", ["*.py"])
        exclude_patterns = self.config.get("exclude_patterns", [])
        
        all_files = []
        for pattern in patterns:
            files = find_files(context.base_dir, pattern, recursive=True, exclude_patterns=exclude_patterns)
            all_files.extend(files)
        
        return all_files
    
    def _analyze_file(self, file_path: Path) -> List[MCPFunctionInfo]:
        """파일에서 함수 추출 및 분석
        
        Args:
            file_path: 분석할 파일 경로
            
        Returns:
            분석된 함수 정보 목록
        """
        self.logger.debug(f"파일 분석 중: {file_path}")
        
        # 파일 파싱
        ast_tree, errors = safe_parse_python(file_path)
        if errors:
            self.logger.warning(f"파일 파싱 중 오류 발생: {file_path} - {errors}")
            if not ast_tree:
                return []
        
        # 파일 내용 읽기
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
        except Exception as e:
            self.logger.error(f"파일 읽기 오류: {file_path} - {e}")
            file_content = ""
        
        # 함수 추출
        functions = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                # 클래스 내 메서드 처리
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        func_info = self._extract_function_info(item, file_path, file_content, class_name=node.name)
                        if func_info:
                            functions.append(func_info)
            elif isinstance(node, ast.FunctionDef):
                # 일반 함수 처리
                func_info = self._extract_function_info(node, file_path, file_content)
                if func_info:
                    functions.append(func_info)
        
        # 함수 상세 분석
        for func in functions:
            self._analyze_function(func)
        
        return functions
    
    def _extract_function_info(self, node: ast.FunctionDef, file_path: Path, 
                              file_content: str, class_name: Optional[str] = None) -> Optional[MCPFunctionInfo]:
        """함수 정보 추출
        
        Args:
            node: 함수 정의 AST 노드
            file_path: 파일 경로
            file_content: 파일 내용
            class_name: 클래스 이름 (메서드인 경우)
            
        Returns:
            함수 정보 객체 또는 None
        """
        # 소스 코드 위치 확인
        if not hasattr(node, 'lineno'):
            return None
            
        line_number = node.lineno
        end_line_number = 0
        
        # 끝 라인 번호 추정
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                end_line_number = max(end_line_number, child.lineno)
        
        # 소스 코드 추출
        lines = file_content.split('\n')
        try:
            source_lines = lines[line_number-1:end_line_number]
            source_code = '\n'.join(source_lines)
        except:
            # 소스 코드 추출 실패 시 AST에서 복원
            source_code = ast.unparse(node)
        
        # 최소 함수 크기 필터
        min_size = self.config.get("min_function_size", 3)
        if len(source_code.split('\n')) < min_size:
            return None
        
        # 함수 정보 생성
        func_info = MCPFunctionInfo(
            name=node.name,
            file_path=file_path,
            line_number=line_number,
            end_line_number=end_line_number,
            source_code=source_code,
            ast_node=node,
            class_name=class_name
        )
        
        return func_info
    
    def _analyze_function(self, func: MCPFunctionInfo) -> None:
        """함수 상세 분석
        
        Args:
            func: 분석할 함수 정보
        """
        # 복잡도 분석
        if func.ast_node:
            func.complexity = calculate_function_complexity(func.ast_node)
        
        # 에러 처리 분석
        if self.config.get("analyze_error_handling", True):
            func.analyze_error_handling()
        
        # 입력 검증 분석
        if self.config.get("analyze_validation", True):
            func.analyze_validation()
        
        # MCP 호환성 검사
        if self.config.get("check_mcp_compatibility", True):
            func.check_mcp_handler()
            func.check_mcp_core_function()
        
        # 함수 호출 추출
        if self.config.get("extract_calls", True):
            func.extract_function_calls()
        
        # 테스트 파일 검색 (향후 구현)
        if self.config.get("find_tests", True):
            func.has_tests = False  # 향후 구현
    
    def _generate_summary(self, functions: List[MCPFunctionInfo]) -> Dict[str, Any]:
        """함수 분석 결과 요약 생성
        
        Args:
            functions: 분석된 함수 목록
            
        Returns:
            요약 정보 딕셔너리
        """
        if not functions:
            return {
                "function_count": 0
            }
        
        # 기본 집계
        total = len(functions)
        public_count = sum(1 for f in functions if f.is_public)
        private_count = total - public_count
        method_count = sum(1 for f in functions if f.is_method)
        standalone_count = total - method_count
        
        # 문서화 및 타입 힌트
        documented_count = sum(1 for f in functions if f.has_docstring)
        typed_count = sum(1 for f in functions if f.has_type_hints)
        
        # 에러 처리
        error_handling_count = sum(1 for f in functions if f.has_error_handling)
        
        # MCP 관련
        mcp_handler_count = sum(1 for f in functions if f.is_mcp_handler)
        mcp_core_count = sum(1 for f in functions if f.is_mcp_core_function)
        
        # 복잡도 평균
        avg_cyclomatic = sum(f.complexity.get("cyclomatic_complexity", 1) for f in functions) / total if total else 0
        avg_nesting = sum(f.complexity.get("max_nesting", 0) for f in functions) / total if total else 0
        avg_params = sum(len(f.parameters) for f in functions) / total if total else 0
        
        # 복잡도 분포
        high_complexity = sum(1 for f in functions if f.complexity.get("cyclomatic_complexity", 1) > 10)
        medium_complexity = sum(1 for f in functions if 5 < f.complexity.get("cyclomatic_complexity", 1) <= 10)
        low_complexity = sum(1 for f in functions if f.complexity.get("cyclomatic_complexity", 1) <= 5)
        
        return {
            "function_count": total,
            "public_count": public_count,
            "private_count": private_count,
            "method_count": method_count,
            "standalone_count": standalone_count,
            "documented_count": documented_count,
            "typed_count": typed_count,
            "error_handling_count": error_handling_count,
            "mcp_handler_count": mcp_handler_count,
            "mcp_core_count": mcp_core_count,
            "avg_cyclomatic": avg_cyclomatic,
            "avg_nesting": avg_nesting,
            "avg_params": avg_params,
            "complexity_distribution": {
                "high": high_complexity,
                "medium": medium_complexity,
                "low": low_complexity
            },
            "documentation_ratio": documented_count / total if total else 0,
            "type_hint_ratio": typed_count / total if total else 0,
            "error_handling_ratio": error_handling_count / total if total else 0
        }
    
    def _calculate_metrics(self, functions: List[MCPFunctionInfo]) -> Dict[str, Any]:
        """성능 지표 계산
        
        Args:
            functions: 분석된 함수 목록
            
        Returns:
            측정 지표 딕셔너리
        """
        if not functions:
            return {}
            
        total = len(functions)
        
        # 기본 지표
        metrics = {
            "function_count": total,
            "avg_function_length": sum(len(f.source_code.split('\n')) for f in functions) / total,
            "avg_complexity": sum(f.complexity.get("cyclomatic_complexity", 1) for f in functions) / total,
            "avg_params": sum(len(f.parameters) for f in functions) / total,
            "documentation_ratio": sum(1 for f in functions if f.has_docstring) / total,
            "type_hint_ratio": sum(1 for f in functions if f.has_type_hints) / total,
            "error_handling_ratio": sum(1 for f in functions if f.has_error_handling) / total,
            "validation_avg": sum(f.validation_score for f in functions) / total,
            "mcp_handler_ratio": sum(1 for f in functions if f.is_mcp_handler) / total,
            "mcp_core_ratio": sum(1 for f in functions if f.is_mcp_core_function) / total
        }
        
        # 품질 점수 계산
        quality_components = {
            "documentation": metrics["documentation_ratio"] * 0.25,
            "type_hints": metrics["type_hint_ratio"] * 0.15,
            "error_handling": metrics["error_handling_ratio"] * 0.25,
            "validation": metrics["validation_avg"] * 0.15,
            "complexity": (1.0 - min(1.0, (metrics["avg_complexity"] - 1) / 9)) * 0.2
        }
        
        metrics["quality_score"] = sum(quality_components.values())
        metrics["quality_components"] = quality_components
        
        return metrics


if __name__ == "__main__":
    # 모듈 테스트 코드
    LoggingManager.setup(log_level='DEBUG')
    
    # 현재 디렉토리 분석
    analyzer = FunctionAnalyzer()
    result = analyzer.run(base_dir=Path("."))
    
    # 결과 출력
    print("\n함수 분석 결과:")
    summary = result.results.get("summary", {})
    
    print(f"총 함수 수: {summary.get('function_count', 0)}")
    print(f"공개 함수: {summary.get('public_count', 0)}")
    print(f"비공개 함수: {summary.get('private_count', 0)}")
    print(f"문서화 비율: {summary.get('documentation_ratio', 0):.1%}")
    print(f"타입 힌트 비율: {summary.get('type_hint_ratio', 0):.1%}")
    print(f"에러 처리 비율: {summary.get('error_handling_ratio', 0):.1%}")
    print(f"평균 복잡도: {summary.get('avg_cyclomatic', 0):.2f}")
    print(f"MCP 핸들러 함수: {summary.get('mcp_handler_count', 0)}")
    print(f"MCP 핵심 함수: {summary.get('mcp_core_count', 0)}")
    
    # 품질 점수
    quality_score = result.metrics.get("quality_score", 0)
    print(f"\n전체 품질 점수: {quality_score:.2f} / 1.0")
    
    # 개별 함수 정보 샘플
    functions = result.results.get("functions", [])
    if functions:
        print("\n함수 샘플:")
        for i, func in enumerate(functions[:3]):
            print(f"  {i+1}. {func['name']} ({func['file_path']}:{func['line_number']})")
            print(f"     복잡도: {func['complexity'].get('cyclomatic_complexity', 0)}")
            print(f"     문서화: {'예' if func['has_docstring'] else '아니오'}")
            print(f"     에러 처리: {'예' if func['has_error_handling'] else '아니오'}")
            print(f"     MCP 핸들러: {'예' if func['is_mcp_handler'] else '아니오'}")
            print()
