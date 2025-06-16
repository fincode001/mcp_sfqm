#!/usr/bin/env python3
"""
MCP-SFQM: 품질 검증 및 자동 수정 모듈
===================================

MCP 서버의 품질을 검증하고 일반적인 오류를 자동으로 수정합니다.
첨부파일 기준의 실전적 품질 관리 체계를 구현합니다.
"""

import ast
import os
import re
import logging
import autopep8
import black
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess
import tempfile
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class QualityValidator:
    """MCP 서버 품질 검증 및 자동 수정"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backup_dir = config.get('core.backup_dir', tempfile.gettempdir())
        
    async def validate_and_fix(self, server_path: str) -> Dict[str, Any]:
        """서버 품질 검증 및 자동 수정
        
        Args:
            server_path: 서버 파일 경로
            
        Returns:
            검증 및 수정 결과
        """
        self.logger.info(f"품질 검증 시작: {server_path}")
        
        result = {
            "server_path": server_path,
            "timestamp": datetime.now().isoformat(),
            "original_issues": [],
            "fixes_applied": [],
            "errors_fixed": 0,
            "warnings_fixed": 0,
            "final_score": 0,
            "backup_created": False
        }
        
        try:
            # 백업 생성
            backup_path = await self._create_backup(server_path)
            result["backup_path"] = backup_path
            result["backup_created"] = True
            
            # 원본 파일 읽기
            with open(server_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            
            # 초기 문제점 분석
            initial_issues = await self._analyze_issues(original_content)
            result["original_issues"] = initial_issues
            
            # 단계별 수정 적용
            fixed_content = original_content
            
            # 1. 문법 오류 수정
            fixed_content, syntax_fixes = await self._fix_syntax_errors(fixed_content)
            result["fixes_applied"].extend(syntax_fixes)
            
            # 2. 코드 스타일 수정
            fixed_content, style_fixes = await self._fix_code_style(fixed_content)
            result["fixes_applied"].extend(style_fixes)
            
            # 3. MCP 관련 이슈 수정
            fixed_content, mcp_fixes = await self._fix_mcp_issues(fixed_content)
            result["fixes_applied"].extend(mcp_fixes)
            
            # 4. 품질 개선
            fixed_content, quality_fixes = await self._improve_quality(fixed_content)
            result["fixes_applied"].extend(quality_fixes)
            
            # 수정된 파일 저장
            if fixed_content != original_content:
                with open(server_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                self.logger.info(f"파일 수정 완료: {server_path}")
            
            # 최종 검증
            final_issues = await self._analyze_issues(fixed_content)
            
            # 통계 계산
            result["errors_fixed"] = len(initial_issues.get("errors", [])) - len(final_issues.get("errors", []))
            result["warnings_fixed"] = len(initial_issues.get("warnings", [])) - len(final_issues.get("warnings", []))
            result["final_score"] = await self._calculate_quality_score(fixed_content)
            
            self.logger.info(f"품질 검증 완료: {server_path} (점수: {result['final_score']}/100)")
            
        except Exception as e:
            self.logger.error(f"품질 검증 실패 {server_path}: {e}")
            result["error"] = str(e)
            
            # 오류 발생 시 백업에서 복원
            if result["backup_created"]:
                await self._restore_from_backup(server_path, result["backup_path"])
        
        return result
    
    async def _create_backup(self, server_path: str) -> str:
        """백업 파일 생성"""
        os.makedirs(self.backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(server_path)}.backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        shutil.copy2(server_path, backup_path)
        self.logger.debug(f"백업 생성: {backup_path}")
        
        return backup_path
    
    async def _restore_from_backup(self, server_path: str, backup_path: str) -> None:
        """백업에서 복원"""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, server_path)
            self.logger.info(f"백업에서 복원: {server_path}")
    
    async def _analyze_issues(self, content: str) -> Dict[str, List]:
        """코드 이슈 분석"""
        issues = {
            "errors": [],
            "warnings": [],
            "style_issues": []
        }
        
        # 문법 오류 검사
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues["errors"].append({
                "type": "SyntaxError",
                "message": str(e),
                "line": e.lineno,
                "severity": "error"
            })
        
        # 일반적인 문제 패턴 검사
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 긴 줄 체크
            if len(line) > 120:
                issues["style_issues"].append({
                    "type": "LineTooLong",
                    "message": f"Line too long ({len(line)} > 120 characters)",
                    "line": i,
                    "severity": "warning"
                })
            
            # 들여쓰기 문제
            if line and not line.startswith((' ', '\t')) and ':' in line and not line.strip().endswith(':'):
                stripped = line.strip()
                if any(stripped.startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except']):
                    if not stripped.endswith(':'):
                        issues["warnings"].append({
                            "type": "MissingColon",
                            "message": "Missing colon after control structure",
                            "line": i,
                            "severity": "warning"
                        })
            
            # 안전하지 않은 패턴
            unsafe_patterns = [
                (r'eval\s*\(', "Use of eval() function"),
                (r'exec\s*\(', "Use of exec() function"),
                (r'input\s*\(', "Use of input() function in server code"),
                (r'print\s*\(', "Use of print() instead of logging")
            ]
            
            for pattern, message in unsafe_patterns:
                if re.search(pattern, line):
                    issues["warnings"].append({
                        "type": "UnsafePattern",
                        "message": message,
                        "line": i,
                        "severity": "warning"
                    })
        
        return issues
    
    async def _fix_syntax_errors(self, content: str) -> Tuple[str, List[Dict]]:
        """문법 오류 자동 수정"""
        fixes = []
        lines = content.split('\n')
        
        # 일반적인 문법 오류 패턴 수정
        for i, line in enumerate(lines):
            original_line = line
            
            # 누락된 콜론 추가
            if line.strip() and not line.strip().endswith(':'):
                stripped = line.strip()
                control_keywords = ['def ', 'class ', 'if ', 'elif ', 'else', 'for ', 'while ', 'try', 'except', 'finally', 'with ']
                
                for keyword in control_keywords:
                    if stripped.startswith(keyword) and '(' in stripped and ')' in stripped:
                        if not stripped.endswith(':'):
                            lines[i] = line + ':'
                            fixes.append({
                                "type": "AddMissingColon",
                                "line": i + 1,
                                "original": original_line,
                                "fixed": lines[i]
                            })
                            break
            
            # 잘못된 들여쓰기 수정 (간단한 경우만)
            if line and line[0] not in [' ', '\t'] and line.strip().startswith(('return ', 'break', 'continue', 'pass')):
                # 이전 줄의 들여쓰기 확인
                if i > 0:
                    prev_line = lines[i-1]
                    if prev_line.strip().endswith(':'):
                        indent = len(prev_line) - len(prev_line.lstrip()) + 4
                        lines[i] = ' ' * indent + line.strip()
                        fixes.append({
                            "type": "FixIndentation",
                            "line": i + 1,
                            "original": original_line,
                            "fixed": lines[i]
                        })
        
        return '\n'.join(lines), fixes
    
    async def _fix_code_style(self, content: str) -> Tuple[str, List[Dict]]:
        """코드 스타일 자동 수정"""
        fixes = []
        
        try:
            # autopep8로 PEP 8 스타일 수정
            fixed_content = autopep8.fix_code(
                content,
                options={
                    'max_line_length': 120,
                    'aggressive': 1,
                    'experimental': True
                }
            )
            
            if fixed_content != content:
                fixes.append({
                    "type": "PEP8StyleFix",
                    "description": "Applied PEP 8 style formatting"
                })
            
            # Black 포매터 적용 (옵션)
            if self.config.get('validators.style.use_black', False):
                try:
                    black_fixed = black.format_str(fixed_content, mode=black.FileMode())
                    if black_fixed != fixed_content:
                        fixed_content = black_fixed
                        fixes.append({
                            "type": "BlackFormatting",
                            "description": "Applied Black code formatting"
                        })
                except Exception as e:
                    self.logger.warning(f"Black formatting failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Code style fixing failed: {e}")
            fixed_content = content
        
        return fixed_content, fixes
    
    async def _fix_mcp_issues(self, content: str) -> Tuple[str, List[Dict]]:
        """MCP 관련 이슈 수정"""
        fixes = []
        lines = content.split('\n')
        
        # MCP import 확인 및 추가
        has_mcp_import = any('import mcp' in line or 'from mcp' in line for line in lines)
        
        if not has_mcp_import:
            # 기본 MCP import 추가
            import_line = "from mcp import types, server"
            
            # import 섹션 찾기
            import_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_index = i + 1
            
            lines.insert(import_index, import_line)
            fixes.append({
                "type": "AddMCPImport",
                "description": "Added missing MCP import",
                "line": import_index + 1
            })
        
        # 로깅 추가 권장
        has_logging = any('import logging' in line or 'from logging' in line for line in lines)
        
        if not has_logging:
            logging_lines = [
                "import logging",
                "",
                "logger = logging.getLogger(__name__)"
            ]
            
            # import 섹션 끝에 추가
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_end = i + 1
            
            for j, log_line in enumerate(logging_lines):
                lines.insert(import_end + j, log_line)
            
            fixes.append({
                "type": "AddLogging",
                "description": "Added logging setup",
                "lines_added": len(logging_lines)
            })
        
        return '\n'.join(lines), fixes
    
    async def _improve_quality(self, content: str) -> Tuple[str, List[Dict]]:
        """코드 품질 개선"""
        fixes = []
        lines = content.split('\n')
        
        # print 문을 로깅으로 변경
        for i, line in enumerate(lines):
            if 'print(' in line and not line.strip().startswith('#'):
                # 간단한 print 문을 logger로 변경
                if line.strip().startswith('print('):
                    original_line = line
                    indent = len(line) - len(line.lstrip())
                    print_content = line.strip()[6:-1]  # print( ... ) 에서 내용 추출
                    
                    if print_content.startswith('"') or print_content.startswith("'"):
                        new_line = ' ' * indent + f'logger.info({print_content})'
                        lines[i] = new_line
                        fixes.append({
                            "type": "PrintToLogging",
                            "line": i + 1,
                            "original": original_line,
                            "fixed": new_line
                        })
        
        # 예외 처리 개선
        for i, line in enumerate(lines):
            if line.strip() == 'except:':
                original_line = line
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + 'except Exception as e:'
                lines[i] = new_line
                
                # 다음 줄에 로깅 추가
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if 'pass' in next_line.strip():
                        log_line = ' ' * (indent + 4) + 'logger.error(f"Error: {e}")'
                        lines[i + 1] = log_line
                
                fixes.append({
                    "type": "ImproveExceptionHandling",
                    "line": i + 1,
                    "original": original_line,
                    "fixed": new_line
                })
        
        return '\n'.join(lines), fixes
    
    async def _calculate_quality_score(self, content: str) -> int:
        """품질 점수 계산 (0-100)"""
        score = 100
        
        # 문법 검사
        try:
            ast.parse(content)
        except SyntaxError:
            score -= 40
        
        # 스타일 검사
        issues = await self._analyze_issues(content)
        score -= len(issues.get("style_issues", [])) * 2
        score -= len(issues.get("warnings", [])) * 5
        score -= len(issues.get("errors", [])) * 10
        
        # 기본 품질 지표
        lines = content.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if code_lines:
            # 주석 비율
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            comment_ratio = len(comment_lines) / len(code_lines)
            
            if comment_ratio < 0.1:
                score -= 10  # 주석이 너무 적음
            elif comment_ratio > 0.3:
                score += 5   # 적절한 주석
            
            # 함수 길이 체크
            avg_func_length = len(code_lines) // max(1, content.count('def '))
            if avg_func_length > 50:
                score -= 15  # 함수가 너무 김
        
        # MCP 관련 체크
        if 'mcp' in content.lower() or 'server' in content.lower():
            score += 10  # MCP 서버로 인식
        
        return max(0, min(100, score))
    
    async def final_validation(self) -> Dict[str, Any]:
        """최종 전체 검증"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "total_servers_validated": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "average_quality_score": 0,
            "issues_found": [],
            "recommendations": []
        }
        
        # 구현 예정: 전체 서버 디렉토리 스캔 및 검증
        self.logger.info("최종 검증 실행")
        
        return validation
