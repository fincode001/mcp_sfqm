#!/usr/bin/env python3
"""
MCP-SFQM: 서버 분석기 모듈
=========================

실제 MCP 서버 파일을 분석하여 다음을 수행:
- MCP 서버 유효성 검증
- 전문화 등급 평가 (A급/B급/C급/폐기)
- 서버 품질 지표 측정
- 기능 복잡도 및 유지보수성 평가
"""

import ast
import os
import sys
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess
import importlib.util
from datetime import datetime

logger = logging.getLogger(__name__)

class ServerAnalyzer:
    """MCP 서버 분석기"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def analyze_server(self, server_path: str) -> Dict[str, Any]:
        """MCP 서버 파일 완전 분석
        
        Args:
            server_path: 서버 파일 경로
            
        Returns:
            분석 결과 딕셔너리
        """
        self.logger.info(f"서버 분석 시작: {server_path}")
        
        try:
            # 파일 존재 확인
            if not os.path.exists(server_path):
                return {"is_valid_mcp_server": False, "error": "File not found"}
            
            # 파일 읽기
            with open(server_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 기본 정보 추출
            basic_info = self._extract_basic_info(server_path, content)
            
            # MCP 서버 유효성 검증
            mcp_validity = self._validate_mcp_server(content)
            
            # 문법 오류 검사
            syntax_check = self._check_syntax(content, server_path)
            
            # 기능 복잡도 분석
            complexity = self._analyze_complexity(content)
            
            # 의존성 분석
            dependencies = self._analyze_dependencies(content)
            
            # 품질 지표 계산
            quality_metrics = self._calculate_quality_metrics(content)
            
            # 종합 분석 결과
            analysis_result = {
                "file_path": server_path,
                "timestamp": datetime.now().isoformat(),
                "basic_info": basic_info,
                "is_valid_mcp_server": mcp_validity["is_valid"],
                "mcp_details": mcp_validity,
                "syntax_check": syntax_check,
                "complexity": complexity,
                "dependencies": dependencies,
                "quality_metrics": quality_metrics,
                "analysis_score": self._calculate_overall_score(
                    mcp_validity, syntax_check, complexity, quality_metrics
                )
            }
            
            self.logger.info(f"서버 분석 완료: {server_path} (점수: {analysis_result['analysis_score']}/100)")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"서버 분석 실패 {server_path}: {e}")
            return {
                "file_path": server_path,
                "is_valid_mcp_server": False,
                "error": str(e),
                "analysis_score": 0
            }
    
    def _extract_basic_info(self, server_path: str, content: str) -> Dict[str, Any]:
        """기본 파일 정보 추출"""
        return {
            "file_name": os.path.basename(server_path),
            "file_size": len(content),
            "line_count": len(content.split('\n')),
            "modification_time": os.path.getmtime(server_path),
            "has_docstring": '"""' in content or "'''" in content,
            "has_main_block": 'if __name__ == "__main__"' in content
        }
    
    def _validate_mcp_server(self, content: str) -> Dict[str, Any]:
        """MCP 서버 유효성 검증"""
        validation = {
            "is_valid": False,
            "has_mcp_import": False,
            "has_server_class": False,
            "has_tools_definition": False,
            "has_handler_methods": False,
            "mcp_version": None,
            "tool_count": 0,
            "handler_count": 0
        }
        
        # MCP import 확인
        mcp_patterns = [
            r'from mcp import',
            r'import mcp',
            r'from mcp\.',
            r'mcp\.server',
            r'mcp\.Server'
        ]
        
        for pattern in mcp_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                validation["has_mcp_import"] = True
                break
        
        # 서버 클래스 또는 인스턴스 확인
        server_patterns = [
            r'class.*Server',
            r'Server\(',
            r'server\s*=.*Server',
            r'app\s*=.*Server'
        ]
        
        for pattern in server_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                validation["has_server_class"] = True
                break
        
        # 도구 정의 확인
        tool_patterns = [
            r'@app\.tool',
            r'@server\.tool',
            r'\.add_tool',
            r'def\s+\w+.*tool',
            r'tools\s*=',
            r'TOOLS\s*='
        ]
        
        tool_matches = []
        for pattern in tool_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tool_matches.extend(matches)
        
        if tool_matches:
            validation["has_tools_definition"] = True
            validation["tool_count"] = len(tool_matches)
        
        # 핸들러 메소드 확인
        handler_patterns = [
            r'async def\s+handle_',
            r'def\s+handle_',
            r'async def\s+\w+_handler',
            r'def\s+\w+_handler'
        ]
        
        handler_matches = []
        for pattern in handler_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            handler_matches.extend(matches)
        
        if handler_matches:
            validation["has_handler_methods"] = True
            validation["handler_count"] = len(handler_matches)
        
        # 전체 유효성 판단
        validation["is_valid"] = (
            validation["has_mcp_import"] and
            (validation["has_server_class"] or validation["has_tools_definition"]) and
            validation["tool_count"] > 0
        )
        
        return validation
    
    def _check_syntax(self, content: str, file_path: str) -> Dict[str, Any]:
        """Python 문법 오류 검사"""
        syntax_result = {
            "is_valid": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # AST로 파싱 시도
            ast.parse(content)
            syntax_result["is_valid"] = True
            
        except SyntaxError as e:
            syntax_result["errors"].append({
                "type": "SyntaxError",
                "message": str(e),
                "line": e.lineno,
                "offset": e.offset
            })
        except Exception as e:
            syntax_result["errors"].append({
                "type": "ParseError",
                "message": str(e)
            })
        
        # 일반적인 문제 패턴 검사
        common_issues = [
            (r'print\s*\(.*\)', "print 문 사용 (로깅 권장)"),
            (r'except\s*:', "bare except 사용 (구체적 예외 처리 권장)"),
            (r'import\s+\*', "wildcard import 사용"),
            (r'eval\s*\(', "eval() 함수 사용 (보안 위험)"),
            (r'exec\s*\(', "exec() 함수 사용 (보안 위험)")
        ]
        
        for pattern, warning in common_issues:
            if re.search(pattern, content):
                syntax_result["warnings"].append(warning)
        
        return syntax_result
    
    def _analyze_complexity(self, content: str) -> Dict[str, Any]:
        """코드 복잡도 분석"""
        complexity = {
            "cyclomatic_complexity": 0,
            "cognitive_complexity": 0,
            "function_count": 0,
            "class_count": 0,
            "max_nesting_depth": 0,
            "average_function_length": 0
        }
        
        try:
            tree = ast.parse(content)
            
            # 함수 및 클래스 개수
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity["function_count"] += 1
                elif isinstance(node, ast.ClassDef):
                    complexity["class_count"] += 1
            
            # 순환 복잡도 계산 (간단한 근사치)
            control_structures = [
                r'\bif\b', r'\bwhile\b', r'\bfor\b', r'\btry\b', 
                r'\bexcept\b', r'\belif\b', r'\bwith\b'
            ]
            
            for pattern in control_structures:
                complexity["cyclomatic_complexity"] += len(re.findall(pattern, content))
            
            # 중첩 깊이 추정
            max_indent = 0
            for line in content.split('\n'):
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent // 4)  # 4칸 들여쓰기 기준
            
            complexity["max_nesting_depth"] = max_indent
            
            # 평균 함수 길이
            if complexity["function_count"] > 0:
                total_lines = len([line for line in content.split('\n') if line.strip()])
                complexity["average_function_length"] = total_lines // complexity["function_count"]
        
        except Exception as e:
            logger.warning(f"복잡도 분석 실패: {e}")
        
        return complexity
    
    def _analyze_dependencies(self, content: str) -> Dict[str, Any]:
        """의존성 분석"""
        dependencies = {
            "standard_library": [],
            "third_party": [],
            "local_imports": [],
            "mcp_related": [],
            "total_imports": 0
        }
        
        # import 문 추출
        import_patterns = [
            r'^import\s+([^\s]+)',
            r'^from\s+([^\s]+)\s+import'
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    dependencies["total_imports"] += 1
                    
                    if 'mcp' in module.lower():
                        dependencies["mcp_related"].append(module)
                    elif module in ['os', 'sys', 'json', 'logging', 're', 'datetime', 'pathlib']:
                        dependencies["standard_library"].append(module)
                    elif module.startswith('.'):
                        dependencies["local_imports"].append(module)
                    else:
                        dependencies["third_party"].append(module)
        
        return dependencies
    
    def _calculate_quality_metrics(self, content: str) -> Dict[str, Any]:
        """품질 지표 계산"""
        metrics = {
            "lines_of_code": 0,
            "comment_ratio": 0.0,
            "docstring_coverage": 0.0,
            "maintainability_index": 0,
            "code_duplication": 0.0
        }
        
        lines = content.split('\n')
        code_lines = 0
        comment_lines = 0
        docstring_lines = 0
        
        in_docstring = False
        docstring_char = None
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                continue
                
            # 독스트링 체크
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    docstring_char = stripped[:3]
                    docstring_lines += 1
                    continue
            else:
                docstring_lines += 1
                if docstring_char in stripped:
                    in_docstring = False
                continue
            
            # 주석 체크
            if stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
        
        metrics["lines_of_code"] = code_lines
        
        if code_lines > 0:
            metrics["comment_ratio"] = comment_lines / (code_lines + comment_lines)
            metrics["docstring_coverage"] = docstring_lines / code_lines
        
        # 유지보수성 지수 (간단한 근사치)
        if code_lines > 0:
            complexity_penalty = min(50, code_lines / 10)  # 코드 길이에 따른 패널티
            comment_bonus = metrics["comment_ratio"] * 20  # 주석 비율에 따른 보너스
            metrics["maintainability_index"] = max(0, 100 - complexity_penalty + comment_bonus)
        
        return metrics
    
    def _calculate_overall_score(self, mcp_validity: Dict, syntax_check: Dict, 
                                complexity: Dict, quality_metrics: Dict) -> int:
        """종합 점수 계산 (0-100)"""
        score = 0
        
        # MCP 유효성 (40점)
        if mcp_validity["is_valid"]:
            score += 40
            # 도구 개수에 따른 추가 점수
            score += min(10, mcp_validity.get("tool_count", 0) * 2)
        
        # 문법 정확성 (20점)
        if syntax_check["is_valid"]:
            score += 20
        
        # 복잡도 (20점)
        complexity_score = 20
        if complexity.get("cyclomatic_complexity", 0) > 20:
            complexity_score -= 10
        if complexity.get("max_nesting_depth", 0) > 5:
            complexity_score -= 5
        score += max(0, complexity_score)
        
        # 품질 지표 (20점)
        maintainability = quality_metrics.get("maintainability_index", 0)
        score += int(maintainability * 0.2)
        
        return min(100, max(0, score))
    
    async def evaluate_specialization_grade(self, server_path: str, functions: List[Dict], 
                                          categories: Dict) -> Dict[str, Any]:
        """전문화 등급 평가 (A급/B급/C급/폐기)"""
        
        # 서버 분석 실행
        analysis = await self.analyze_server(server_path)
        
        grade_info = {
            "server_path": server_path,
            "grade": "C",  # 기본값
            "score": analysis.get("analysis_score", 0),
            "reasons": [],
            "recommendations": []
        }
        
        score = analysis.get("analysis_score", 0)
        
        # 등급 결정 로직
        if score >= 80:
            grade_info["grade"] = "A"
            grade_info["reasons"].append("높은 코드 품질과 MCP 표준 준수")
        elif score >= 60:
            grade_info["grade"] = "B"
            grade_info["reasons"].append("양호한 품질, 일부 개선 필요")
        elif score >= 40:
            grade_info["grade"] = "C"
            grade_info["reasons"].append("기본 기능은 동작하나 품질 개선 필요")
        else:
            grade_info["grade"] = "DISCARD"
            grade_info["reasons"].append("품질이 너무 낮음, 폐기 권장")
        
        # 추가 검증 사항
        if not analysis.get("is_valid_mcp_server", False):
            grade_info["grade"] = "DISCARD"
            grade_info["reasons"].append("유효한 MCP 서버가 아님")
        
        if not analysis.get("syntax_check", {}).get("is_valid", False):
            grade_info["grade"] = "DISCARD"
            grade_info["reasons"].append("문법 오류 존재")
        
        # 권장사항 생성
        if grade_info["grade"] in ["B", "C"]:
            if analysis.get("quality_metrics", {}).get("comment_ratio", 0) < 0.1:
                grade_info["recommendations"].append("주석 및 문서화 개선")
            
            if analysis.get("complexity", {}).get("cyclomatic_complexity", 0) > 15:
                grade_info["recommendations"].append("코드 복잡도 감소")
        
        return grade_info
