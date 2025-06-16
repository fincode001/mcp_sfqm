#!/usr/bin/env python3
"""
MCP-SFQM: Logging Management Module
-----------------------------------
로깅 관리 모듈: 시스템 전체 로깅 설정 및 관리

이 모듈은 MCP-SFQM 시스템 전체에서 일관된 로깅을 위한 설정 및 유틸리티를 제공합니다.
다양한 출력 대상(파일, 콘솔, 원격 서버 등)과 형식을 지원하며, 컴포넌트별 로깅 레벨을 설정할 수 있습니다.

기능:
- 중앙 집중식 로깅 설정
- 다양한 출력 형식 (텍스트, JSON)
- 컴포넌트별 로깅 레벨 설정
- 로그 파일 로테이션
- 로그 필터링 및 색상 지원

사용법:
    from mcp_sfqm.core.logging_manager import LoggingManager
    
    # 로깅 설정 초기화
    LoggingManager.setup(log_level='INFO', log_file='mcp_sfqm.log')
    
    # 로거 인스턴스 가져오기
    logger = LoggingManager.get_logger(__name__)
    
    # 로깅
    logger.info("분석 시작")
    logger.debug("상세 정보: %s", details)
    logger.error("오류 발생", exc_info=True)
"""

import os
import sys
import logging
import logging.handlers
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
from datetime import datetime
import traceback
import threading
import atexit

# 컬러 로깅을 위한 ANSI 색상 코드
COLORS = {
    'RESET': '\033[0m',
    'BLACK': '\033[30m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
}

# 로그 레벨별 색상 매핑
LEVEL_COLORS = {
    'DEBUG': COLORS['BLUE'],
    'INFO': COLORS['GREEN'],
    'WARNING': COLORS['YELLOW'],
    'ERROR': COLORS['RED'],
    'CRITICAL': COLORS['BOLD'] + COLORS['RED'],
}

class ColoredFormatter(logging.Formatter):
    """컬러 지원 로그 포매터"""
    
    def __init__(self, fmt: str = None, datefmt: str = None, style: str = '%', use_colors: bool = True):
        """컬러 포매터 초기화
        
        Args:
            fmt: 로그 포맷 문자열
            datefmt: 날짜 포맷 문자열
            style: 포맷 스타일 ('%', '{', '$')
            use_colors: 색상 사용 여부
        """
        super().__init__(fmt, datefmt, style)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드 포맷
        
        Args:
            record: 로그 레코드
            
        Returns:
            포맷된 로그 문자열
        """
        # 원본 메시지 저장
        original_msg = record.msg
        levelname = record.levelname
        
        # 색상 적용 (Windows는 ANSI 색상 지원이 제한적)
        if self.use_colors and not (sys.platform == 'win32' and not 'ANSICON' in os.environ):
            color = LEVEL_COLORS.get(levelname, COLORS['RESET'])
            record.levelname = f"{color}{levelname}{COLORS['RESET']}"
            
            # 오류 메시지에 색상 적용 (문자열인 경우만)
            if record.levelno >= logging.ERROR and isinstance(record.msg, str):
                record.msg = f"{color}{record.msg}{COLORS['RESET']}"
        
        # 상위 클래스의 format 호출
        formatted = super().format(record)
        
        # 원본 메시지 복원
        record.msg = original_msg
        record.levelname = levelname
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포매터"""
    
    def __init__(self, include_timestamp: bool = True, include_hostname: bool = True):
        """JSON 포매터 초기화
        
        Args:
            include_timestamp: 타임스탬프 포함 여부
            include_hostname: 호스트명 포함 여부
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_hostname = include_hostname
        
        # 호스트명 캐싱
        self.hostname = None
        if include_hostname:
            import socket
            try:
                self.hostname = socket.gethostname()
            except:
                self.hostname = 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형식으로 포맷
        
        Args:
            record: 로그 레코드
            
        Returns:
            JSON 형식 로그 문자열
        """
        log_data = {
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'lineno': record.lineno,
        }
        
        # 타임스탬프 추가
        if self.include_timestamp:
            log_data['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        
        # 호스트명 추가
        if self.include_hostname and self.hostname:
            log_data['hostname'] = self.hostname
        
        # 예외 정보 추가
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 추가 컨텍스트 정보
        if hasattr(record, 'context') and record.context:
            log_data['context'] = record.context
        
        # JSON으로 변환
        return json.dumps(log_data)


class ContextAdapter(logging.LoggerAdapter):
    """컨텍스트 정보를 로그에 추가하는 어댑터"""
    
    def process(self, msg, kwargs):
        """로그 메시지 처리
        
        Args:
            msg: 로그 메시지
            kwargs: 로깅 함수에 전달된 키워드 인자
            
        Returns:
            처리된 메시지와 키워드 인자
        """
        # 컨텍스트 정보 추출
        context = kwargs.pop('context', {})
        if context:
            # 로그 레코드에 컨텍스트 추가
            extra = kwargs.get('extra', {})
            extra['context'] = context
            kwargs['extra'] = extra
        
        return msg, kwargs


class LogQueueHandler(logging.handlers.QueueHandler):
    """로그 큐 핸들러 (비동기 로깅)"""
    
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """로그 레코드 준비
        
        Args:
            record: 원본 로그 레코드
            
        Returns:
            처리된 로그 레코드
        """
        # 복사본 생성 및 예외 정보 보존
        copy = super().prepare(record)
        
        # 예외 정보가 있는 경우 복사
        if record.exc_info:
            copy.exc_info = record.exc_info
            copy.exc_text = record.exc_text
        
        # 컨텍스트 정보가 있는 경우 복사
        if hasattr(record, 'context'):
            copy.context = record.context
        
        return copy


class LoggingManager:
    """MCP-SFQM 로깅 관리자"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls) -> 'LoggingManager':
        """싱글톤 인스턴스 반환
        
        Returns:
            LoggingManager 인스턴스
        """
        if cls._instance is None:
            cls._instance = LoggingManager()
        return cls._instance
    
    def __init__(self):
        """로깅 관리자 초기화"""
        # 이미 초기화되었는지 확인
        if LoggingManager._instance is not None:
            raise RuntimeError("LoggingManager는 싱글톤 클래스입니다. get_instance()를 사용하세요.")
        
        # 로깅 설정 상태
        self._initialized = False
        
        # 비동기 로깅 설정
        self._log_queue = None
        self._queue_listener = None
        
        # 컴포넌트별 로거 캐시
        self._loggers = {}
        
        # 기본 로깅 설정
        self._default_log_level = logging.INFO
        self._default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self._default_date_format = '%Y-%m-%d %H:%M:%S'
        
        # 컴포넌트별 로그 레벨
        self._component_levels = {}
    
    @classmethod
    def setup(cls, log_level: Union[str, int] = 'INFO', log_file: Optional[str] = None,
              format_str: Optional[str] = None, date_format: Optional[str] = None,
              component_levels: Optional[Dict[str, str]] = None, use_colors: bool = True,
              json_output: bool = False, async_logging: bool = True, 
              max_bytes: int = 10*1024*1024, backup_count: int = 5,
              config_file: Optional[str] = None) -> None:
        """로깅 설정 초기화
        
        Args:
            log_level: 기본 로그 레벨 (이름 또는 숫자)
            log_file: 로그 파일 경로 (None이면 콘솔만 사용)
            format_str: 로그 포맷 문자열
            date_format: 날짜 포맷 문자열
            component_levels: 컴포넌트별 로그 레벨 (예: {'analyzers': 'DEBUG'})
            use_colors: 콘솔 출력에 색상 사용 여부
            json_output: JSON 형식으로 출력 여부
            async_logging: 비동기 로깅 사용 여부
            max_bytes: 로그 파일 최대 크기 (바이트)
            backup_count: 보관할 로그 파일 수
            config_file: 로깅 설정 파일 경로 (YAML/JSON)
        """
        instance = cls.get_instance()
        
        # 설정 파일에서 로드
        if config_file and os.path.exists(config_file):
            instance._load_config(config_file)
        
        # 파라미터로 전달된 설정 적용
        if isinstance(log_level, str):
            level = getattr(logging, log_level.upper(), None)
            if level is None:
                raise ValueError(f"잘못된 로그 레벨: {log_level}")
            instance._default_log_level = level
        else:
            instance._default_log_level = log_level
            
        if format_str:
            instance._default_format = format_str
            
        if date_format:
            instance._default_date_format = date_format
            
        if component_levels:
            instance._component_levels = component_levels
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(instance._default_log_level)
        
        # 기존 핸들러 제거
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        
        # 로그 포매터 생성
        if json_output:
            formatter = JSONFormatter()
        else:
            formatter = ColoredFormatter(
                fmt=instance._default_format,
                datefmt=instance._default_date_format,
                use_colors=use_colors
            )
        
        # 핸들러 생성
        handlers = []
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # 파일 핸들러 (지정된 경우)
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # 비동기 로깅 설정
        if async_logging:
            instance._setup_async_logging(handlers)
        else:
            # 핸들러 직접 추가
            for handler in handlers:
                root_logger.addHandler(handler)
        
        # 컴포넌트별 로그 레벨 설정
        for component, level in instance._component_levels.items():
            component_logger = logging.getLogger(component)
            level_value = getattr(logging, level.upper(), None)
            if level_value is not None:
                component_logger.setLevel(level_value)
        
        instance._initialized = True
        logging.info("로깅 시스템이 초기화되었습니다.")
    
    def _load_config(self, config_file: str) -> None:
        """설정 파일에서 로깅 설정 로드
        
        Args:
            config_file: 설정 파일 경로 (YAML/JSON)
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError(f"지원되지 않는 설정 파일 형식: {config_file}")
            
            # 설정 적용
            if 'logging' in config:
                logging_config = config['logging']
                
                if 'level' in logging_config:
                    self._default_log_level = getattr(logging, logging_config['level'].upper())
                    
                if 'format' in logging_config:
                    self._default_format = logging_config['format']
                    
                if 'date_format' in logging_config:
                    self._default_date_format = logging_config['date_format']
                    
                if 'components' in logging_config:
                    self._component_levels = logging_config['components']
            
        except Exception as e:
            logging.warning(f"로깅 설정 파일을 로드하는 중 오류 발생: {e}")
    
    def _setup_async_logging(self, handlers: List[logging.Handler]) -> None:
        """비동기 로깅 설정
        
        Args:
            handlers: 로그 핸들러 목록
        """
        # 이전 리스너 정리
        if self._queue_listener is not None:
            self._queue_listener.stop()
        
        # 로그 큐 설정
        import queue
        self._log_queue = queue.Queue(-1)  # 무제한 큐
        
        # 큐 핸들러
        queue_handler = LogQueueHandler(self._log_queue)
        import logging as log_module
        root_logger = log_module.getLogger()
        root_logger.addHandler(queue_handler)        
        # 큐 리스너
        import logging.handlers
        self._queue_listener = logging.handlers.QueueListener(
            self._log_queue, *handlers, respect_handler_level=True
        )
        self._queue_listener.start()
        
        # 프로그램 종료 시 리스너 정리
        atexit.register(self._cleanup)
    
    def _cleanup(self) -> None:
        """로깅 리소스 정리"""
        if self._queue_listener is not None:
            self._queue_listener.stop()
            self._queue_listener = None
    
    @classmethod
    def get_logger(cls, name: str, with_context: bool = False) -> Union[logging.Logger, ContextAdapter]:
        """지정된 이름의 로거 반환
        
        Args:
            name: 로거 이름 (일반적으로 __name__)
            with_context: 컨텍스트 어댑터 사용 여부
            
        Returns:
            로거 또는 컨텍스트 어댑터
        """
        instance = cls.get_instance()
        
        # 아직 초기화되지 않은 경우 기본 설정으로 초기화
        if not instance._initialized:
            cls.setup()
        
        # 캐시된 로거 확인
        cache_key = f"{name}:{with_context}"
        if cache_key in instance._loggers:
            return instance._loggers[cache_key]
        
        # 새 로거 생성
        logger = logging.getLogger(name)
        
        # 컴포넌트 로그 레벨 설정
        for component, level in instance._component_levels.items():
            if name.startswith(component):
                level_value = getattr(logging, level.upper(), None)
                if level_value is not None:
                    logger.setLevel(level_value)
                break
        
        # 컨텍스트 어댑터 사용 시
        if with_context:
            logger = ContextAdapter(logger, {})
        
        # 캐시에 저장
        instance._loggers[cache_key] = logger
        return logger
    
    @classmethod
    def set_level(cls, name: str, level: Union[str, int]) -> None:
        """특정 로거의 로그 레벨 설정
        
        Args:
            name: 로거 이름
            level: 설정할 로그 레벨 (이름 또는 숫자)
        """
        if isinstance(level, str):
            level_value = getattr(logging, level.upper(), None)
            if level_value is None:
                raise ValueError(f"잘못된 로그 레벨: {level}")
        else:
            level_value = level
        
        logger = logging.getLogger(name)
        logger.setLevel(level_value)
        
        # 컴포넌트 레벨 업데이트
        instance = cls.get_instance()
        instance._component_levels[name] = logging.getLevelName(level_value)
    
    @classmethod
    def create_child_logger(cls, parent_name: str, child_name: str, with_context: bool = False) -> Union[logging.Logger, ContextAdapter]:
        """부모 로거에서 자식 로거 생성
        
        Args:
            parent_name: 부모 로거 이름
            child_name: 자식 로거 이름
            with_context: 컨텍스트 어댑터 사용 여부
            
        Returns:
            자식 로거
        """
        full_name = f"{parent_name}.{child_name}"
        return cls.get_logger(full_name, with_context)
    
    @classmethod
    def log_with_context(cls, logger: logging.Logger, level: int, msg: str, context: Dict[str, Any], *args, **kwargs) -> None:
        """컨텍스트 정보와 함께 로그 기록
        
        Args:
            logger: 로거 인스턴스
            level: 로그 레벨
            msg: 로그 메시지
            context: 컨텍스트 정보
            *args: 포맷 인자
            **kwargs: 추가 키워드 인자
        """
        if isinstance(logger, ContextAdapter):
            logger.log(level, msg, *args, context=context, **kwargs)
        else:
            # 일반 로거에 컨텍스트 추가
            extra = kwargs.get('extra', {})
            extra['context'] = context
            kwargs['extra'] = extra
            logger.log(level, msg, *args, **kwargs)


if __name__ == "__main__":
    # 모듈 테스트 코드
    # 로깅 설정
    LoggingManager.setup(
        log_level='DEBUG',
        log_file='test_logging.log',
        use_colors=True,
        component_levels={
            'test.component1': 'INFO',
            'test.component2': 'DEBUG'
        }
    )
    
    # 테스트 로거
    logger = LoggingManager.get_logger(__name__)
    logger.debug("디버그 메시지")
    logger.info("정보 메시지")
    logger.warning("경고 메시지")
    logger.error("오류 메시지")
    
    # 컨텍스트 로거
    context_logger = LoggingManager.get_logger("test.context", with_context=True)
    context_logger.info("컨텍스트 정보와 함께", context={
        'user': 'admin',
        'operation': 'test',
        'timestamp': datetime.now().isoformat()
    })
    
    # 컴포넌트 로거
    comp1 = LoggingManager.get_logger("test.component1")
    comp1.debug("이 메시지는 보이지 않습니다 (INFO 레벨)")
    comp1.info("컴포넌트1 메시지")
    
    comp2 = LoggingManager.get_logger("test.component2")
    comp2.debug("컴포넌트2 디버그 메시지 (보입니다)")
    
    # 예외 로깅
    try:
        x = 1 / 0
    except Exception as e:
        logger.error("계산 중 오류 발생", exc_info=True)
        
        # 컨텍스트와 함께 예외 로깅
        context_logger.error(
            "컨텍스트와 함께 예외 발생",
            context={'operation': 'division', 'value': 0},
            exc_info=True
        )
