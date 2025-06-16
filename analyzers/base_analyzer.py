#!/usr/bin/env python3
"""
MCP-SFQM: Base Analyzer Module
------------------------------
기본 분석기 모듈: 모든 분석기의 기본 클래스 정의

이 모듈은 MCP 서버 분석 엔진의 기본 클래스와 인터페이스를 정의합니다.
모든 특화된 분석기는 이 기본 클래스를 상속받아 구현됩니다.

특징:
- 일관된 분석기 인터페이스
- 분석 컨텍스트 관리
- 진행 상황 및 이벤트 추적
- 결과 저장 및 로드
- 비동기 및 병렬 분석 지원

사용법:
    from mcp_sfqm.analyzers.base_analyzer import BaseAnalyzer, AnalysisContext
    
    class MyCustomAnalyzer(BaseAnalyzer):
        def __init__(self, config=None):
            super().__init__(config=config)
            
        def analyze(self, context):
            # 구현...
            pass
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 상위 패키지 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_sfqm.core.exceptions import AnalysisError
from mcp_sfqm.core.logging_manager import LoggingManager
from mcp_sfqm.core.utils import to_json, load_json, ensure_directory

# 로거 설정
logger = LoggingManager.get_logger(__name__, with_context=True)


@dataclass
class AnalysisContext:
    """분석 컨텍스트
    
    분석 중 필요한 데이터, 설정, 상태 등을 담는 컨텍스트 객체
    """
    
    # 기본 정보
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 입력 설정
    base_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    target_files: List[Path] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 분석 상태
    status: str = "initialized"  # initialized, running, completed, failed
    progress: float = 0.0  # 0.0 ~ 1.0
    current_stage: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # 분석 결과
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 캐시 및 중간 데이터
    cache: Dict[str, Any] = field(default_factory=dict)
    temp_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """초기화 후처리"""
        # 경로 객체 변환
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
            
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        # 경로 리스트 변환
        self.target_files = [Path(f) if isinstance(f, str) else f for f in self.target_files]
    
    def start(self) -> None:
        """분석 시작 시간 기록"""
        self.status = "running"
        self.start_time = time.time()
    
    def complete(self) -> None:
        """분석 완료 시간 기록"""
        self.status = "completed"
        self.end_time = time.time()
        self.progress = 1.0
    
    def fail(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """분석 실패 상태 설정
        
        Args:
            error_message: 오류 메시지
            error_details: 추가 오류 상세 정보
        """
        self.status = "failed"
        self.end_time = time.time()
        
        error = {
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        if error_details:
            error.update(error_details)
            
        self.errors.append(error)
    
    def add_error(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """오류 정보 추가
        
        Args:
            message: 오류 메시지
            details: 추가 오류 상세 정보
        """
        error = {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            error.update(details)
            
        self.errors.append(error)
    
    def add_warning(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """경고 정보 추가
        
        Args:
            message: 경고 메시지
            details: 추가 경고 상세 정보
        """
        warning = {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            warning.update(details)
            
        self.warnings.append(warning)
    
    def update_progress(self, progress: float, stage: Optional[str] = None) -> None:
        """진행 상황 업데이트
        
        Args:
            progress: 진행률 (0.0 ~ 1.0)
            stage: 현재 진행 단계 설명
        """
        self.progress = max(0.0, min(1.0, progress))  # 0.0 ~ 1.0 범위로 제한
        
        if stage:
            self.current_stage = stage
    
    def add_result(self, key: str, value: Any) -> None:
        """분석 결과 추가
        
        Args:
            key: 결과 키
            value: 결과 값
        """
        self.results[key] = value
    
    def add_metric(self, key: str, value: Any) -> None:
        """측정 지표 추가
        
        Args:
            key: 지표 키
            value: 지표 값
        """
        self.metrics[key] = value
    
    def cache_data(self, key: str, value: Any) -> None:
        """데이터 캐싱
        
        Args:
            key: 캐시 키
            value: 캐시할 값
        """
        self.cache[key] = value
    
    def get_cached_data(self, key: str, default: Any = None) -> Any:
        """캐시된 데이터 획득
        
        Args:
            key: 캐시 키
            default: 캐시 키가 없을 경우 반환할 기본값
            
        Returns:
            캐시된 값 또는 기본값
        """
        return self.cache.get(key, default)
    
    def elapsed_time(self) -> float:
        """분석에 소요된 시간 계산 (초)
        
        Returns:
            소요 시간 (초)
        """
        if not self.start_time:
            return 0.0
            
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """컨텍스트를 딕셔너리로 변환
        
        Returns:
            딕셔너리 표현
        """
        return asdict(self)
    
    def to_json(self, pretty: bool = True) -> str:
        """컨텍스트를 JSON 문자열로 변환
        
        Args:
            pretty: 들여쓰기 적용 여부
            
        Returns:
            JSON 문자열
        """
        return to_json(self.to_dict(), pretty)
    
    def save(self, file_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """컨텍스트를 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로 (None이면 기본 경로 사용)
            
        Returns:
            저장된 파일 경로 또는 실패 시 None
        """
        if not file_path and not self.output_dir:
            logger.error("저장할 파일 경로가 지정되지 않았습니다.")
            return None
            
        if not file_path:
            # 기본 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_context_{timestamp}.json"
            file_path = self.output_dir / filename
        else:
            file_path = Path(file_path)
            
        # 디렉토리 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
            logger.debug(f"분석 컨텍스트 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"분석 컨텍스트 저장 실패: {e}")
            return None
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Optional['AnalysisContext']:
        """파일에서 컨텍스트 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            로드된 컨텍스트 또는 실패 시 None
        """
        try:
            data = load_json(file_path)
            if not data:
                return None
                
            # Path 객체 복원
            if 'base_dir' in data and data['base_dir']:
                data['base_dir'] = Path(data['base_dir'])
                
            if 'output_dir' in data and data['output_dir']:
                data['output_dir'] = Path(data['output_dir'])
                
            if 'target_files' in data:
                data['target_files'] = [Path(f) for f in data['target_files']]
                
            # 데이터 클래스 인스턴스 생성
            context = cls(**data)
            logger.debug(f"분석 컨텍스트 로드 완료: {file_path}")
            return context
            
        except Exception as e:
            logger.error(f"분석 컨텍스트 로드 실패: {file_path} - {e}")
            return None


class AnalysisEvent:
    """분석 이벤트 타입 정의"""
    STARTED = "analysis_started"
    COMPLETED = "analysis_completed"
    FAILED = "analysis_failed"
    PROGRESS_UPDATED = "progress_updated"
    RESULT_ADDED = "result_added"
    ERROR_ADDED = "error_added"
    WARNING_ADDED = "warning_added"


class BaseAnalyzer(ABC):
    """기본 분석기 추상 클래스
    
    모든 분석기의 기본 클래스로, 공통 기능을 제공합니다.
    각 특화된 분석기는 이 클래스를 상속받아 구현해야 합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """기본 분석기 초기화
        
        Args:
            config: 분석기 설정
        """
        self.config = config or {}
        self.logger = LoggingManager.get_logger(f"{__name__}.{self.__class__.__name__}", with_context=True)
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
    
    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalysisContext:
        """분석 실행
        
        이 메서드는 각 분석기에서 구현해야 합니다.
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            업데이트된 분석 컨텍스트
        """
        pass
    
    def prepare(self, context: AnalysisContext) -> AnalysisContext:
        """분석 전 준비 작업
        
        기본 구현은 단순히 컨텍스트를 시작 상태로 설정합니다.
        필요에 따라 자식 클래스에서 오버라이드할 수 있습니다.
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            준비된 분석 컨텍스트
        """
        # 출력 디렉토리 확인 및 생성
        if context.output_dir:
            ensure_directory(context.output_dir)
        
        # 분석 시작 설정
        context.start()
        self._trigger_event(AnalysisEvent.STARTED, context)
        
        self.logger.info("분석 준비 완료", context={"analyzer": self.__class__.__name__})
        return context
    
    def finalize(self, context: AnalysisContext) -> AnalysisContext:
        """분석 후 마무리 작업
        
        기본 구현은 단순히 컨텍스트를 완료 상태로 설정합니다.
        필요에 따라 자식 클래스에서 오버라이드할 수 있습니다.
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            마무리된 분석 컨텍스트
        """
        # 오류가 없으면 완료 상태로 설정
        if not context.errors:
            context.complete()
            self._trigger_event(AnalysisEvent.COMPLETED, context)
            self.logger.info(f"분석 완료 (소요 시간: {context.elapsed_time():.2f}초)", 
                            context={"analyzer": self.__class__.__name__})
        else:
            # 이미 실패 상태가 아니면 설정
            if context.status != "failed":
                context.status = "failed"
                context.end_time = time.time()
                self._trigger_event(AnalysisEvent.FAILED, context)
                
            error_count = len(context.errors)
            self.logger.error(f"분석 실패 (오류: {error_count}개)", 
                             context={"analyzer": self.__class__.__name__})
        
        # 결과 저장
        if context.output_dir:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = context.output_dir / f"analysis_result_{self.__class__.__name__}_{timestamp}.json"
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(to_json(context.results))
                    
                self.logger.info(f"분석 결과 저장 완료: {result_file}", 
                                context={"analyzer": self.__class__.__name__})
                
                # 전체 컨텍스트도 저장
                context.save()
                
            except Exception as e:
                self.logger.error(f"분석 결과 저장 실패: {e}", 
                                 context={"analyzer": self.__class__.__name__})
        
        return context
    
    def run(self, context: Optional[AnalysisContext] = None, **kwargs) -> AnalysisContext:
        """분석 실행 (준비, 분석, 마무리 포함)
        
        Args:
            context: 분석 컨텍스트 (None이면 새로 생성)
            **kwargs: 컨텍스트 생성 시 전달할 추가 인자
            
        Returns:
            완료된 분석 컨텍스트
        """
        # 컨텍스트가 없으면 새로 생성
        if context is None:
            context = self._create_context(**kwargs)
        
        try:
            # 준비
            context = self.prepare(context)
            
            # 분석
            context = self.analyze(context)
            
            # 마무리
            context = self.finalize(context)
            
        except Exception as e:
            self.logger.error(f"분석 중 예외 발생: {e}", exc_info=True, 
                             context={"analyzer": self.__class__.__name__})
            
            # 예외 정보 컨텍스트에 추가
            context.fail(str(e), {
                "exception_type": type(e).__name__,
                "traceback": self.logger.findCaller()
            })
            
            # 이벤트 트리거
            self._trigger_event(AnalysisEvent.FAILED, context)
            
            # 마무리 시도
            try:
                context = self.finalize(context)
            except:
                pass
        
        return context
    
    def _create_context(self, **kwargs) -> AnalysisContext:
        """새 분석 컨텍스트 생성
        
        Args:
            **kwargs: 컨텍스트 생성 시 전달할 추가 인자
            
        Returns:
            새 분석 컨텍스트
        """
        # 기본 설정에서 값 가져오기
        config = kwargs.pop('config', self.config)
        
        # 컨텍스트 생성
        context = AnalysisContext(config=config, **kwargs)
        return context
    
    def add_event_handler(self, event_type: str, handler: Callable[[AnalysisContext], None]) -> None:
        """이벤트 핸들러 추가
        
        Args:
            event_type: 이벤트 타입
            handler: 이벤트 핸들러 함수
        """
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable[[AnalysisContext], None]) -> bool:
        """이벤트 핸들러 제거
        
        Args:
            event_type: 이벤트 타입
            handler: 제거할 이벤트 핸들러 함수
            
        Returns:
            제거 성공 여부
        """
        with self._lock:
            if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)
                return True
        return False
    
    def _trigger_event(self, event_type: str, context: AnalysisContext) -> None:
        """이벤트 트리거
        
        Args:
            event_type: 이벤트 타입
            context: 이벤트와 연관된 분석 컨텍스트
        """
        with self._lock:
            handlers = self._event_handlers.get(event_type, [])
            
        for handler in handlers:
            try:
                handler(context)
            except Exception as e:
                self.logger.error(f"이벤트 핸들러 실행 중 오류: {e}", 
                                 context={"event_type": event_type, "analyzer": self.__class__.__name__})
    
    def update_progress(self, context: AnalysisContext, progress: float, stage: Optional[str] = None) -> None:
        """진행 상황 업데이트 및 이벤트 트리거
        
        Args:
            context: 분석 컨텍스트
            progress: 진행률 (0.0 ~ 1.0)
            stage: 현재 진행 단계 설명
        """
        context.update_progress(progress, stage)
        self._trigger_event(AnalysisEvent.PROGRESS_UPDATED, context)
        
        # 로그 출력 (10% 단위로)
        progress_percent = int(progress * 100)
        if progress_percent % 10 == 0 and progress_percent > 0:
            stage_info = f" - {stage}" if stage else ""
            self.logger.info(f"분석 진행 중: {progress_percent}%{stage_info}", 
                            context={"analyzer": self.__class__.__name__, "progress": progress_percent})
    
    def add_result(self, context: AnalysisContext, key: str, value: Any) -> None:
        """결과 추가 및 이벤트 트리거
        
        Args:
            context: 분석 컨텍스트
            key: 결과 키
            value: 결과 값
        """
        context.add_result(key, value)
        self._trigger_event(AnalysisEvent.RESULT_ADDED, context)
    
    def add_error(self, context: AnalysisContext, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """오류 추가 및 이벤트 트리거
        
        Args:
            context: 분석 컨텍스트
            message: 오류 메시지
            details: 추가 오류 상세 정보
        """
        context.add_error(message, details)
        self._trigger_event(AnalysisEvent.ERROR_ADDED, context)
        
        # 로그 출력
        self.logger.error(message, context={"analyzer": self.__class__.__name__, **(details or {})})
    
    def add_warning(self, context: AnalysisContext, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """경고 추가 및 이벤트 트리거
        
        Args:
            context: 분석 컨텍스트
            message: 경고 메시지
            details: 추가 경고 상세 정보
        """
        context.add_warning(message, details)
        self._trigger_event(AnalysisEvent.WARNING_ADDED, context)
        
        # 로그 출력
        self.logger.warning(message, context={"analyzer": self.__class__.__name__, **(details or {})})
    
    def process_in_parallel(self, items: List[Any], process_func: Callable[[Any], Any], 
                           max_workers: Optional[int] = None, use_processes: bool = False) -> List[Any]:
        """항목 병렬 처리
        
        Args:
            items: 처리할 항목 목록
            process_func: 처리 함수
            max_workers: 최대 작업자 수 (None이면 CPU 코어 수 사용)
            use_processes: 프로세스 사용 여부 (True면 프로세스, False면 스레드)
            
        Returns:
            처리 결과 목록
        """
        if not items:
            return []
        
        # 설정에서 작업자 수 가져오기 (기본값은 CPU 코어 수)
        if max_workers is None:
            max_workers = self.config.get('max_workers', os.cpu_count())
            
        # 항목 수가 작으면 병렬 처리 불필요
        if len(items) <= 1:
            return [process_func(item) for item in items]
            
        # 병렬 처리
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            results = list(executor.map(process_func, items))
            
        return results


if __name__ == "__main__":
    # 모듈 테스트 코드
    LoggingManager.setup(log_level='DEBUG')
    
    # 테스트용 분석기 구현
    class TestAnalyzer(BaseAnalyzer):
        def analyze(self, context):
            # 간단한 분석 시뮬레이션
            self.logger.info("테스트 분석 시작")
            
            # 진행 상황 업데이트
            for i in range(10):
                time.sleep(0.1)
                self.update_progress(context, (i + 1) / 10, f"테스트 단계 {i+1}")
            
            # 결과 추가
            self.add_result(context, "test_value", 42)
            self.add_result(context, "test_list", [1, 2, 3, 4])
            
            # 경고 추가
            self.add_warning(context, "테스트 경고", {"source": "test"})
            
            return context
    
    # 이벤트 핸들러 정의
    def on_progress(context):
        print(f"진행 상황: {context.progress * 100:.0f}% - {context.current_stage}")
        
    def on_complete(context):
        print(f"분석 완료: {len(context.results)} 결과, {len(context.warnings)} 경고")
    
    # 분석 실행
    analyzer = TestAnalyzer()
    analyzer.add_event_handler(AnalysisEvent.PROGRESS_UPDATED, on_progress)
    analyzer.add_event_handler(AnalysisEvent.COMPLETED, on_complete)
    
    # 임시 출력 디렉토리 생성
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 컨텍스트 생성 및 실행
    result = analyzer.run(base_dir=Path("."), output_dir=output_dir)
    
    print("\n결과:")
    print(result.to_json())
