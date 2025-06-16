#!/usr/bin/env python3
"""
MCP-SFQM: Configuration Management Module
-----------------------------------------
설정 관리 모듈: 시스템 전체 설정을 중앙에서 관리하고 모든 구성 요소에 일관된 설정을 제공

기능:
- YAML/JSON 기반 설정 파일 관리
- 환경변수, 명령줄 인자, 설정 파일 우선순위 처리
- 동적 설정 업데이트 및 재로드
- 플러그인별 설정 관리
- 설정 유효성 검증

사용법:
    from mcp_sfqm.core.config import ConfigManager
    
    # 글로벌 설정 인스턴스 획득
    config = ConfigManager.get_instance()
    
    # 설정 값 접근
    threshold = config.get('analyzers.similarity.threshold', default=0.7)
    
    # 설정 값 업데이트
    config.set('reporters.output_dir', '/path/to/reports')
    
    # 전체 설정 다시 로드
    config.reload()
"""

import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import argparse
import tempfile
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """설정 관련 예외"""
    pass

class ConfigManager:
    """MCP-SFQM 설정 관리자"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls, config_file: Optional[str] = None) -> 'ConfigManager':
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = ConfigManager(config_file)
        elif config_file:
            cls._instance.load_config(config_file)
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None):
        """설정 관리자 초기화"""
        self._config: Dict[str, Any] = {}
        self._config_file = config_file
        self._config_dir = None
        self._default_config_loaded = False
        self._env_prefix = "MCP_SFQM_"
        
        # 기본 설정 로드
        self._load_default_config()
        
        # 지정된 설정 파일 로드
        if config_file:
            self.load_config(config_file)
            
        # 환경 변수 적용
        self._apply_environment_variables()
    
    def _load_default_config(self) -> None:
        """기본 설정 로드"""
        default_config = {
            "core": {
                "temp_dir": tempfile.gettempdir(),
                "backup_dir": os.path.join(tempfile.gettempdir(), "mcp_sfqm_backups"),
                "log_level": "INFO",
                "max_workers": os.cpu_count() or 4,
            },
            "analyzers": {
                "similarity": {
                    "threshold": 0.7,
                    "algorithm": "jaccard",
                    "min_tokens": 10,
                },
                "complexity": {
                    "max_cyclomatic": 15,
                    "max_cognitive": 20,
                    "max_maintainability_index": 100,
                },
                "quality": {
                    "min_documentation": 0.3,
                    "min_test_coverage": 0.6,
                    "error_handling_required": True,
                }
            },
            "validators": {
                "syntax": {
                    "enabled": True,
                    "fix_common_errors": True,
                },
                "mcp_spec": {
                    "enabled": True,
                    "required_methods": ["get_model_names", "get_model_metadata", "execute"],
                    "protocol_version": "1.0",
                },
                "api": {
                    "enabled": True,
                    "check_backward_compatibility": True,
                },
                "performance": {
                    "enabled": True,
                    "max_response_time": 1.0,  # seconds
                    "max_memory_usage": 512,   # MB
                }
            },
            "integrators": {
                "backup": {
                    "enabled": True,
                    "keep_backups": 5,
                },
                "merge": {
                    "similarity_threshold": 0.8,
                    "max_merge_candidates": 5,
                    "preserve_comments": True,
                    "auto_resolve_conflicts": True,
                },
                "rollback": {
                    "enabled": True,
                    "auto_rollback_on_error": True,
                }
            },
            "reporters": {
                "output_dir": "./mcp_sfqm_reports",
                "formats": ["text", "json", "html"],
                "include_graphs": True,
                "detailed_function_reports": True,
                "anonymize_paths": False,
            },
            "plugins": {
                "enabled": True,
                "search_paths": ["./plugins"],
                "auto_discover": True,
            }
        }
        
        self._config = default_config
        self._default_config_loaded = True
        logger.debug("Default configuration loaded")
    
    def load_config(self, config_file: str) -> None:
        """설정 파일 로드"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
            
        self._config_dir = config_path.parent
        self._config_file = config_file
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
            else:
                raise ConfigError(f"Unsupported configuration file format: {config_path.suffix}")
                
            # 사용자 설정을 기본 설정에 깊은 병합
            self._deep_merge(self._config, user_config)
            logger.info(f"Configuration loaded from: {config_file}")
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise ConfigError(f"Failed to parse configuration file: {e}")
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """두 설정 딕셔너리를 깊게 병합"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _apply_environment_variables(self) -> None:
        """환경 변수에서 설정 적용"""
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower().replace('_', '.')
                self.set(config_key, self._convert_value(value))
                logger.debug(f"Applied environment variable: {key} -> {config_key}")
    
    def _convert_value(self, value: str) -> Any:
        """문자열 값을 적절한 타입으로 변환"""
        # 불리언 값 처리
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
            
        # 숫자 처리
        try:
            # 정수
            if value.isdigit():
                return int(value)
            # 부동소수점
            return float(value)
        except ValueError:
            # 문자열 반환
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 획득
        
        Args:
            key: 점으로 구분된 설정 키 (예: 'analyzers.similarity.threshold')
            default: 키가 없을 경우 반환할 기본값
            
        Returns:
            설정 값 또는 기본값
        """
        parts = key.split('.')
        config = self._config
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
                
        return config
    
    def set(self, key: str, value: Any) -> None:
        """설정 값 설정
        
        Args:
            key: 점으로 구분된 설정 키 (예: 'analyzers.similarity.threshold')
            value: 설정할 값
        """
        parts = key.split('.')
        config = self._config
        
        # 마지막 부분 전까지 탐색하며 필요한 딕셔너리 생성
        for i, part in enumerate(parts[:-1]):
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
            
        # 마지막 부분에 값 설정
        config[parts[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """전체 설정 딕셔너리 반환"""
        return self._config.copy()
    
    def reload(self) -> None:
        """설정 다시 로드"""
        self._load_default_config()
        if self._config_file:
            self.load_config(self._config_file)
        self._apply_environment_variables()
        logger.info("Configuration reloaded")
    
    def save(self, file_path: Optional[str] = None) -> None:
        """현재 설정을 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로 (지정하지 않으면 기존 파일 사용)
        """
        save_path = file_path or self._config_file
        
        if not save_path:
            raise ConfigError("No configuration file path specified")
            
        save_path = Path(save_path)
        
        try:
            # 백업 파일 생성
            if save_path.exists():
                backup_path = save_path.with_suffix(f"{save_path.suffix}.bak")
                shutil.copy2(save_path, backup_path)
            
            # 새 설정 저장
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2)
            else:
                raise ConfigError(f"Unsupported configuration file format: {save_path.suffix}")
                
            logger.info(f"Configuration saved to: {save_path}")
            
        except (IOError, OSError) as e:
            logger.error(f"Error saving configuration: {e}")
            raise ConfigError(f"Failed to save configuration: {e}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """특정 섹션의 설정 반환
        
        Args:
            section: 최상위 섹션 이름 (예: 'analyzers')
            
        Returns:
            해당 섹션의 설정 딕셔너리
        """
        return self._config.get(section, {}).copy()
    
    def validate(self) -> List[str]:
        """현재 설정의 유효성 검증
        
        Returns:
            오류 메시지 목록 (비어있으면 유효함)
        """
        errors = []
        
        # 필수 설정 검증
        required_sections = ['core', 'analyzers', 'validators', 'integrators', 'reporters']
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        # 경로 설정 검증
        output_dir = self.get('reporters.output_dir')
        if output_dir and not os.path.isabs(output_dir):
            self.set('reporters.output_dir', os.path.abspath(output_dir))
        
        backup_dir = self.get('core.backup_dir')
        if backup_dir and not os.path.isabs(backup_dir):
            self.set('core.backup_dir', os.path.abspath(backup_dir))
        
        # 기타 유효성 검증 로직...
        
        return errors
    
    def from_args(self, args: argparse.Namespace) -> None:
        """명령줄 인자에서 설정 적용
        
        Args:
            args: 파싱된 명령줄 인수
        """
        # 일반적인 명령줄 인수 처리
        arg_mapping = {
            'config_file': None,  # 설정 파일은 이미 처리됨
            'output_dir': 'reporters.output_dir',
            'log_level': 'core.log_level',
            'verbose': None,  # 특별 처리
            'quiet': None,  # 특별 처리
            'max_workers': 'core.max_workers',
            'similarity_threshold': 'analyzers.similarity.threshold',
            'fix_errors': 'validators.syntax.fix_common_errors',
            'generate_report': None,  # 특별 처리
        }
        
        for arg_name, config_key in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                value = getattr(args, arg_name)
                
                if config_key:
                    self.set(config_key, value)
                elif arg_name == 'verbose' and value:
                    self.set('core.log_level', 'DEBUG')
                elif arg_name == 'quiet' and value:
                    self.set('core.log_level', 'WARNING')
                elif arg_name == 'generate_report':
                    self.set('reporters.formats', ['text', 'json', 'html'] if value else ['text'])
        
        logger.debug("Applied configuration from command line arguments")


if __name__ == "__main__":
    # 모듈 테스트 코드
    logging.basicConfig(level=logging.DEBUG)
    
    config = ConfigManager.get_instance()
    
    # 기본 설정 출력
    print("Default similarity threshold:", config.get('analyzers.similarity.threshold'))
    
    # 설정 변경
    config.set('analyzers.similarity.threshold', 0.85)
    print("Updated similarity threshold:", config.get('analyzers.similarity.threshold'))
    
    # 전체 설정 출력
    print("\nEntire configuration:")
    import pprint
    pprint.pprint(config.get_all())
