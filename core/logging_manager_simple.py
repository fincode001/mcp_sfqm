#!/usr/bin/env python3
"""
간소화된 로깅 매니저
"""

import logging
import os
import sys
from datetime import datetime

class LoggingManager:
    """간소화된 로깅 매니저"""
    
    _instance = None
    
    def __init__(self, config=None):
        self.config = config or {}
        self.setup_basic_logging()
    
    @classmethod
    def get_instance(cls, config=None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def setup_basic_logging(self):
        """기본 로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'mcp_sfqm_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
    
    @staticmethod
    def get_logger(name, with_context=False):
        """로거 반환"""
        return logging.getLogger(name)
    
    @staticmethod
    def setup():
        """정적 설정 메소드"""
        pass
