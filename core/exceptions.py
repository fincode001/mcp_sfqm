#!/usr/bin/env python3
"""
MCP-SFQM: Exception Handling Module
-----------------------------------
예외 처리 모듈: 시스템 전체에서 사용되는 예외 클래스 정의

이 모듈은 MCP-SFQM 시스템 전체에서 일관된 예외 처리를 위한 표준 예외 클래스를 제공합니다.
각 기능 영역별로 특화된 예외 클래스를 정의하고, 상세한 오류 정보와 디버깅 컨텍스트를 포함합니다.

사용법:
    from mcp_sfqm.core.exceptions import MCPSFQMError, AnalysisError
    
    try:
        # 코드 실행
    except AnalysisError as e:
        logger.error(f"분석 오류: {e}")
        logger.debug(f"컨텍스트: {e.context}")
"""

from typing import Dict, Any, Optional, List
import traceback
import sys
import json
from pathlib import Path
from datetime import datetime

class MCPSFQMError(Exception):
    """MCP-SFQM 시스템의 기본 예외 클래스"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """기본 예외 초기화
        
        Args:
            message: 오류 메시지
            context: 추가 컨텍스트 정보
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
        
    def __str__(self) -> str:
        """사람이 읽기 쉬운 오류 메시지 반환"""
        return f"{self.message}"
        
    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "traceback": self.traceback
        }
        
    def to_json(self) -> str:
        """예외 정보를 JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=2, default=str)
        
    @staticmethod
    def from_exception(exc: Exception, context: Optional[Dict[str, Any]] = None) -> 'MCPSFQMError':
        """일반 예외에서 MCPSFQMError 생성
        
        Args:
            exc: 원본 예외
            context: 추가 컨텍스트 정보
            
        Returns:
            MCPSFQMError 인스턴스
        """
        return MCPSFQMError(str(exc), context)


# 설정 관련 예외
class ConfigError(MCPSFQMError):
    """설정 관련 예외"""
    pass


# 분석 관련 예외
class AnalysisError(MCPSFQMError):
    """분석 관련 기본 예외"""
    pass

class CodeParsingError(AnalysisError):
    """코드 파싱 오류"""
    
    def __init__(self, message: str, file_path: str, line_number: Optional[int] = None, 
                 code_snippet: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """코드 파싱 오류 초기화
        
        Args:
            message: 오류 메시지
            file_path: 파싱 중 오류가 발생한 파일 경로
            line_number: 오류가 발생한 줄 번호
            code_snippet: 오류가 발생한 코드 부분
            context: 추가 컨텍스트 정보
        """
        self.file_path = file_path
        self.line_number = line_number
        self.code_snippet = code_snippet
        
        # 컨텍스트에 파일 정보 추가
        file_context = {
            "file_path": file_path,
            "line_number": line_number,
            "code_snippet": code_snippet
        }
        
        if context:
            context.update(file_context)
        else:
            context = file_context
            
        super().__init__(message, context)
        
    def __str__(self) -> str:
        """사람이 읽기 쉬운 오류 메시지 반환"""
        location = f"{self.file_path}"
        if self.line_number:
            location += f":{self.line_number}"
            
        return f"파싱 오류 ({location}): {self.message}"


class FunctionAnalysisError(AnalysisError):
    """함수 분석 오류"""
    
    def __init__(self, message: str, function_name: str, file_path: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """함수 분석 오류 초기화
        
        Args:
            message: 오류 메시지
            function_name: 분석 중 오류가 발생한 함수 이름
            file_path: 함수가 정의된 파일 경로
            context: 추가 컨텍스트 정보
        """
        self.function_name = function_name
        self.file_path = file_path
        
        # 컨텍스트에 함수 정보 추가
        func_context = {
            "function_name": function_name
        }
        
        if file_path:
            func_context["file_path"] = file_path
            
        if context:
            context.update(func_context)
        else:
            context = func_context
            
        super().__init__(message, context)
        
    def __str__(self) -> str:
        """사람이 읽기 쉬운 오류 메시지 반환"""
        if self.file_path:
            return f"함수 분석 오류 ({self.function_name} in {self.file_path}): {self.message}"
        else:
            return f"함수 분석 오류 ({self.function_name}): {self.message}"


class SimilarityAnalysisError(AnalysisError):
    """유사도 분석 오류"""
    pass


class DependencyAnalysisError(AnalysisError):
    """의존성 분석 오류"""
    pass


# 검증 관련 예외
class ValidationError(MCPSFQMError):
    """검증 관련 기본 예외"""
    pass

class SyntaxValidationError(ValidationError):
    """구문 검증 오류"""
    
    def __init__(self, message: str, file_path: str, line_number: Optional[int] = None,
                 code_snippet: Optional[str] = None, fix_suggestion: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """구문 검증 오류 초기화
        
        Args:
            message: 오류 메시지
            file_path: 검증 중 오류가 발생한 파일 경로
            line_number: 오류가 발생한 줄 번호
            code_snippet: 오류가 발생한 코드 부분
            fix_suggestion: 수정 제안
            context: 추가 컨텍스트 정보
        """
        self.file_path = file_path
        self.line_number = line_number
        self.code_snippet = code_snippet
        self.fix_suggestion = fix_suggestion
        
        # 컨텍스트에 검증 정보 추가
        validation_context = {
            "file_path": file_path,
            "line_number": line_number,
            "code_snippet": code_snippet
        }
        
        if fix_suggestion:
            validation_context["fix_suggestion"] = fix_suggestion
            
        if context:
            context.update(validation_context)
        else:
            context = validation_context
            
        super().__init__(message, context)
        
    def __str__(self) -> str:
        """사람이 읽기 쉬운 오류 메시지 반환"""
        location = f"{self.file_path}"
        if self.line_number:
            location += f":{self.line_number}"
            
        base_msg = f"구문 오류 ({location}): {self.message}"
        
        if self.fix_suggestion:
            base_msg += f"\n수정 제안: {self.fix_suggestion}"
            
        return base_msg


class MCPSpecValidationError(ValidationError):
    """MCP 명세 검증 오류"""
    
    def __init__(self, message: str, server_name: str, missing_features: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        """MCP 명세 검증 오류 초기화
        
        Args:
            message: 오류 메시지
            server_name: 검증 중 오류가 발생한 서버 이름
            missing_features: 누락된 필수 기능 목록
            context: 추가 컨텍스트 정보
        """
        self.server_name = server_name
        self.missing_features = missing_features or []
        
        # 컨텍스트에 MCP 정보 추가
        mcp_context = {
            "server_name": server_name,
            "missing_features": self.missing_features
        }
            
        if context:
            context.update(mcp_context)
        else:
            context = mcp_context
            
        super().__init__(message, context)
        
    def __str__(self) -> str:
        """사람이 읽기 쉬운 오류 메시지 반환"""
        base_msg = f"MCP 명세 오류 ({self.server_name}): {self.message}"
        
        if self.missing_features:
            missing = ", ".join(self.missing_features)
            base_msg += f"\n누락된 기능: {missing}"
            
        return base_msg


class APIValidationError(ValidationError):
    """API 호환성 검증 오류"""
    pass


class PerformanceValidationError(ValidationError):
    """성능 검증 오류"""
    pass


class SecurityValidationError(ValidationError):
    """보안 검증 오류"""
    pass


# 통합 관련 예외
class IntegrationError(MCPSFQMError):
    """통합 관련 기본 예외"""
    pass

class MergeError(IntegrationError):
    """병합 오류"""
    
    def __init__(self, message: str, source_files: List[str], target_file: str,
                 conflict_details: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None):
        """병합 오류 초기화
        
        Args:
            message: 오류 메시지
            source_files: 병합 소스 파일 목록
            target_file: 병합 대상 파일
            conflict_details: 충돌 상세 정보
            context: 추가 컨텍스트 정보
        """
        self.source_files = source_files
        self.target_file = target_file
        self.conflict_details = conflict_details or {}
        
        # 컨텍스트에 병합 정보 추가
        merge_context = {
            "source_files": source_files,
            "target_file": target_file,
            "conflict_details": self.conflict_details
        }
            
        if context:
            context.update(merge_context)
        else:
            context = merge_context
            
        super().__init__(message, context)
        
    def __str__(self) -> str:
        """사람이 읽기 쉬운 오류 메시지 반환"""
        sources = ", ".join([Path(s).name for s in self.source_files])
        target = Path(self.target_file).name
        
        return f"병합 오류 ({sources} -> {target}): {self.message}"


class SplitError(IntegrationError):
    """분리 오류"""
    pass


class BackupError(IntegrationError):
    """백업 오류"""
    pass


class RollbackError(IntegrationError):
    """롤백 오류"""
    pass


# 보고서 관련 예외
class ReportingError(MCPSFQMError):
    """보고서 관련 기본 예외"""
    pass

class ReportGenerationError(ReportingError):
    """보고서 생성 오류"""
    pass


class VisualizationError(ReportingError):
    """시각화 오류"""
    pass


# 플러그인 관련 예외
class PluginError(MCPSFQMError):
    """플러그인 관련 기본 예외"""
    pass

class PluginLoadError(PluginError):
    """플러그인 로드 오류"""
    pass


class PluginExecutionError(PluginError):
    """플러그인 실행 오류"""
    pass


# 유틸리티 함수
def handle_exception(exc: Exception, logger, exit_on_error: bool = False) -> None:
    """예외 처리 및 로깅 유틸리티
    
    Args:
        exc: 발생한 예외
        logger: 로깅에 사용할 로거
        exit_on_error: 오류 발생 시 프로그램 종료 여부
    """
    if isinstance(exc, MCPSFQMError):
        logger.error(str(exc))
        logger.debug(f"예외 컨텍스트: {exc.context}")
        logger.debug(f"트레이스백: {exc.traceback}")
    else:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        logger.error(error_msg)
        logger.debug(f"트레이스백: {traceback.format_exc()}")
    
    if exit_on_error:
        logger.critical("치명적 오류로 인해 프로그램을 종료합니다.")
        sys.exit(1)


if __name__ == "__main__":
    # 모듈 테스트 코드
    import logging
    
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # 테스트 예외 발생
        raise CodeParsingError(
            message="예상치 못한 토큰 발견", 
            file_path="/path/to/file.py", 
            line_number=42, 
            code_snippet="def invalid_func()"
        )
    except MCPSFQMError as e:
        handle_exception(e, logger)
        print("\n예외 JSON 표현:")
        print(e.to_json())
