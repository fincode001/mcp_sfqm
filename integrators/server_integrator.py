#!/usr/bin/env python3
"""
MCP-SFQM: 서버 통합 관리자
=========================

MCP 서버의 통합/분리를 실행하고 관리합니다.
첨부파일 기준의 실전적 통합 전략을 구현합니다.
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import tempfile
import asyncio

logger = logging.getLogger(__name__)

class ServerIntegrator:
    """MCP 서버 통합/분리 관리자"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backup_dir = config.get('core.backup_dir', tempfile.gettempdir())
        self.integration_plan = None
        
    async def create_integration_plan(self, graded_servers: Dict[str, Dict]) -> Dict[str, Any]:
        """통합/분리 계획 수립
        
        Args:
            graded_servers: 등급이 매겨진 서버 딕셔너리
            
        Returns:
            통합/분리 계획
        """
        self.logger.info("통합/분리 계획 수립 시작")
        
        plan = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "conservative",  # conservative, aggressive, balanced
            "actions": [],
            "target_servers": [],
            "backup_required": True,
            "estimated_duration": 0,
            "risk_level": "medium"
        }
        
        # 등급별 서버 분류
        grade_groups = {"A": [], "B": [], "C": [], "DISCARD": []}
        
        for server_path, grade_info in graded_servers.items():
            grade = grade_info.get("grade", "C")
            grade_groups[grade].append({
                "path": server_path,
                "info": grade_info
            })
        
        # 통합/분리 전략 결정
        plan["actions"] = await self._plan_integration_actions(grade_groups)
        plan["target_servers"] = [action["target"] for action in plan["actions"]]
        plan["estimated_duration"] = len(plan["actions"]) * 30  # 초 단위
        
        # 리스크 평가
        discard_ratio = len(grade_groups["DISCARD"]) / max(1, sum(len(g) for g in grade_groups.values()))
        if discard_ratio > 0.5:
            plan["risk_level"] = "high"
        elif discard_ratio < 0.2:
            plan["risk_level"] = "low"
        
        self.integration_plan = plan
        self.logger.info(f"통합 계획 수립 완료: {len(plan['actions'])}개 작업, 위험도: {plan['risk_level']}")
        
        return plan
    
    async def _plan_integration_actions(self, grade_groups: Dict[str, List]) -> List[Dict]:
        """구체적인 통합/분리 작업 계획"""
        actions = []
        
        # 1. DISCARD 등급 서버 아카이브
        for server in grade_groups["DISCARD"]:
            actions.append({
                "type": "archive",
                "target": server["path"],
                "reason": "Low quality - marked for discard",
                "priority": 1
            })
        
        # 2. C급 서버 개선 또는 통합
        c_grade_servers = grade_groups["C"]
        if len(c_grade_servers) > 5:  # C급 서버가 많으면 통합 고려
            # 기능별로 그룹화하여 통합
            function_groups = await self._group_servers_by_function(c_grade_servers)
            
            for group_name, servers in function_groups.items():
                if len(servers) > 1:
                    actions.append({
                        "type": "merge",
                        "target": [s["path"] for s in servers],
                        "output": f"integrated_{group_name}_server.py",
                        "reason": f"Merge {len(servers)} C-grade servers with similar functions",
                        "priority": 2
                    })
        
        # 3. B급 서버 개선
        for server in grade_groups["B"]:
            actions.append({
                "type": "improve",
                "target": server["path"],
                "improvements": ["add_documentation", "optimize_performance", "add_error_handling"],
                "reason": "Upgrade B-grade server to A-grade",
                "priority": 3
            })
        
        # 4. A급 서버 보존 및 최적화
        for server in grade_groups["A"]:
            actions.append({
                "type": "preserve",
                "target": server["path"],
                "optimizations": ["performance_tuning", "documentation_update"],
                "reason": "Maintain and optimize A-grade server",
                "priority": 4
            })
        
        # 우선순위로 정렬
        actions.sort(key=lambda x: x["priority"])
        
        return actions
    
    async def _group_servers_by_function(self, servers: List[Dict]) -> Dict[str, List]:
        """서버를 기능별로 그룹화"""
        function_groups = {
            "file_management": [],
            "data_processing": [],
            "web_scraping": [],
            "api_integration": [],
            "utility": [],
            "other": []
        }
        
        # 간단한 키워드 기반 분류
        keywords = {
            "file_management": ["file", "directory", "path", "folder", "document"],
            "data_processing": ["data", "process", "transform", "analyze", "calculate"],
            "web_scraping": ["web", "scrape", "crawl", "html", "url", "request"],
            "api_integration": ["api", "rest", "endpoint", "service", "client"],
            "utility": ["util", "helper", "tool", "common", "misc"]
        }
        
        for server in servers:
            server_path = server["path"].lower()
            server_name = os.path.basename(server_path)
            
            assigned = False
            for category, category_keywords in keywords.items():
                if any(keyword in server_name for keyword in category_keywords):
                    function_groups[category].append(server)
                    assigned = True
                    break
            
            if not assigned:
                function_groups["other"].append(server)
        
        # 빈 그룹 제거
        return {k: v for k, v in function_groups.items() if v}
    
    async def create_backup(self) -> str:
        """전체 서버 백업 생성"""
        self.logger.info("전체 서버 백업 생성 중...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = os.path.join(self.backup_dir, f"mcp_servers_backup_{timestamp}")
        
        os.makedirs(backup_root, exist_ok=True)
        
        # 현재 서버 디렉토리 전체 백업
        server_pool_path = self.config.get('core.server_pool_path', './mcp_servers')
        if os.path.exists(server_pool_path):
            shutil.copytree(server_pool_path, os.path.join(backup_root, "servers"))
        
        # 백업 메타데이터 저장
        backup_info = {
            "timestamp": timestamp,
            "backup_path": backup_root,
            "original_path": server_pool_path,
            "integration_plan": self.integration_plan
        }
        
        with open(os.path.join(backup_root, "backup_info.json"), 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"백업 완료: {backup_root}")
        return backup_root
    
    async def execute_integration(self) -> Dict[str, Any]:
        """통합/분리 실행"""
        if not self.integration_plan:
            raise ValueError("Integration plan not created")
        
        self.logger.info("통합/분리 실행 시작")
        
        execution_result = {
            "start_time": datetime.now().isoformat(),
            "actions_executed": [],
            "actions_failed": [],
            "total_actions": len(self.integration_plan["actions"]),
            "success_rate": 0
        }
        
        for action in self.integration_plan["actions"]:
            try:
                result = await self._execute_action(action)
                execution_result["actions_executed"].append({
                    "action": action,
                    "result": result,
                    "status": "success"
                })
                
            except Exception as e:
                self.logger.error(f"Action failed: {action['type']} - {e}")
                execution_result["actions_failed"].append({
                    "action": action,
                    "error": str(e),
                    "status": "failed"
                })
        
        execution_result["end_time"] = datetime.now().isoformat()
        execution_result["success_rate"] = len(execution_result["actions_executed"]) / max(1, execution_result["total_actions"])
        
        self.logger.info(f"통합/분리 실행 완료: {execution_result['success_rate']:.2%} 성공률")
        
        return execution_result
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """개별 작업 실행"""
        action_type = action["type"]
        
        if action_type == "archive":
            return await self._archive_server(action["target"])
        elif action_type == "merge":
            return await self._merge_servers(action["target"], action["output"])
        elif action_type == "improve":
            return await self._improve_server(action["target"], action["improvements"])
        elif action_type == "preserve":
            return await self._preserve_server(action["target"], action["optimizations"])
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    async def _archive_server(self, server_path: str) -> Dict[str, Any]:
        """서버 아카이브"""
        archive_dir = os.path.join(self.backup_dir, "archived_servers")
        os.makedirs(archive_dir, exist_ok=True)
        
        server_name = os.path.basename(server_path)
        archive_path = os.path.join(archive_dir, f"{server_name}.archived")
        
        if os.path.exists(server_path):
            shutil.move(server_path, archive_path)
            
        return {
            "original_path": server_path,
            "archive_path": archive_path,
            "action": "archived"
        }
    
    async def _merge_servers(self, server_paths: List[str], output_name: str) -> Dict[str, Any]:
        """서버 통합"""
        self.logger.info(f"서버 통합 시작: {len(server_paths)}개 서버 -> {output_name}")
        
        merged_content = await self._create_merged_server_content(server_paths)
        
        # 출력 경로 결정
        output_dir = os.path.dirname(server_paths[0])
        output_path = os.path.join(output_dir, output_name)
        
        # 통합된 서버 파일 작성
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        
        # 원본 파일들을 백업으로 이동
        backup_paths = []
        for server_path in server_paths:
            if os.path.exists(server_path):
                backup_path = f"{server_path}.merged_backup"
                shutil.move(server_path, backup_path)
                backup_paths.append(backup_path)
        
        return {
            "merged_servers": server_paths,
            "output_path": output_path,
            "backup_paths": backup_paths,
            "action": "merged"
        }
    
    async def _create_merged_server_content(self, server_paths: List[str]) -> str:
        """통합된 서버 내용 생성"""
        merged_parts = [
            '#!/usr/bin/env python3',
            '"""',
            'Integrated MCP Server',
            '=====================',
            '',
            'This server was created by merging the following servers:',
        ]
        
        # 원본 서버 목록 추가
        for path in server_paths:
            merged_parts.append(f'- {os.path.basename(path)}')
        
        merged_parts.extend([
            '',
            f'Created on: {datetime.now().isoformat()}',
            'Generated by: MCP-SFQM Integration System',
            '"""',
            '',
            'import logging',
            'from mcp import types, server',
            'from typing import Any, Dict, List',
            '',
            'logger = logging.getLogger(__name__)',
            '',
            '# Initialize MCP server',
            'app = server.Server("integrated-mcp-server")',
            ''
        ])
        
        # 각 서버의 주요 함수들을 추출하여 통합
        for i, server_path in enumerate(server_paths):
            if os.path.exists(server_path):
                try:
                    with open(server_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    merged_parts.append(f'# ======== Functions from {os.path.basename(server_path)} ========')
                    
                    # 함수 정의 추출 (간단한 정규식 기반)
                    import re
                    functions = re.findall(r'(async def \w+.*?(?=\n(?:async )?def|\n(?:class|\Z))|def \w+.*?(?=\n(?:async )?def|\n(?:class|\Z)))', content, re.DOTALL)
                    
                    for func in functions:
                        if 'handle_' in func or '@app.tool' in func:
                            # 함수 이름에 원본 서버 식별자 추가
                            func_with_prefix = func.replace('def ', f'def server{i+1}_')
                            func_with_prefix = func_with_prefix.replace('async def ', f'async def server{i+1}_')
                            merged_parts.append(func_with_prefix)
                            merged_parts.append('')
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract functions from {server_path}: {e}")
        
        # 메인 실행 부분 추가
        merged_parts.extend([
            '',
            'if __name__ == "__main__":',
            '    import asyncio',
            '    from mcp.server.stdio import stdio_server',
            '    ',
            '    async def main():',
            '        async with stdio_server() as (read_stream, write_stream):',
            '            await app.run(',
            '                read_stream,',
            '                write_stream,',
            '                app.create_initialization_options()',
            '            )',
            '    ',
            '    asyncio.run(main())'
        ])
        
        return '\n'.join(merged_parts)
    
    async def _improve_server(self, server_path: str, improvements: List[str]) -> Dict[str, Any]:
        """서버 개선"""
        applied_improvements = []
        
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"Server not found: {server_path}")
        
        with open(server_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # 개선사항 적용
        for improvement in improvements:
            if improvement == "add_documentation":
                content = await self._add_documentation(content)
                applied_improvements.append("add_documentation")
            
            elif improvement == "optimize_performance":
                content = await self._optimize_performance(content)
                applied_improvements.append("optimize_performance")
            
            elif improvement == "add_error_handling":
                content = await self._add_error_handling(content)
                applied_improvements.append("add_error_handling")
        
        # 변경사항이 있으면 파일 업데이트
        if content != original_content:
            with open(server_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return {
            "server_path": server_path,
            "improvements_applied": applied_improvements,
            "action": "improved"
        }
    
    async def _add_documentation(self, content: str) -> str:
        """문서화 추가"""
        lines = content.split('\n')
        
        # 함수에 독스트링 추가
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                # 다음 줄이 독스트링이 아니면 추가
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith('"""'):
                    func_name = line.strip().split('(')[0].replace('def ', '').replace('async ', '')
                    indent = len(line) - len(line.lstrip())
                    docstring = f'{" " * (indent + 4)}"""TODO: Add documentation for {func_name}"""'
                    lines.insert(i + 1, docstring)
        
        return '\n'.join(lines)
    
    async def _optimize_performance(self, content: str) -> str:
        """성능 최적화"""
        # 간단한 최적화: 불필요한 import 제거, 캐싱 힌트 추가 등
        lines = content.split('\n')
        
        # TODO: 더 정교한 성능 최적화 로직 구현
        return content
    
    async def _add_error_handling(self, content: str) -> str:
        """에러 처리 추가"""
        lines = content.split('\n')
        
        # try-except 블록이 없는 함수에 기본 에러 처리 추가
        for i, line in enumerate(lines):
            if (line.strip().startswith('def ') or line.strip().startswith('async def ')) and 'try:' not in content[i:i+20]:
                # 함수 내용에 기본 try-except 구조 추가 권장 주석
                indent = len(line) - len(line.lstrip())
                comment = f'{" " * (indent + 4)}# TODO: Add proper error handling with try-except'
                lines.insert(i + 1, comment)
        
        return '\n'.join(lines)
    
    async def _preserve_server(self, server_path: str, optimizations: List[str]) -> Dict[str, Any]:
        """A급 서버 보존 및 최적화"""
        return {
            "server_path": server_path,
            "optimizations_planned": optimizations,
            "action": "preserved"
        }
    
    async def rollback(self, backup_path: str) -> Dict[str, Any]:
        """백업에서 롤백"""
        self.logger.info(f"백업에서 롤백 시작: {backup_path}")
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # 백업 정보 로드
        backup_info_path = os.path.join(backup_path, "backup_info.json")
        if os.path.exists(backup_info_path):
            with open(backup_info_path, 'r', encoding='utf-8') as f:
                backup_info = json.load(f)
            
            original_path = backup_info.get("original_path")
            backup_servers_path = os.path.join(backup_path, "servers")
            
            if os.path.exists(backup_servers_path) and original_path:
                # 현재 디렉토리 백업
                if os.path.exists(original_path):
                    rollback_backup = f"{original_path}_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.move(original_path, rollback_backup)
                
                # 백업에서 복원
                shutil.copytree(backup_servers_path, original_path)
                
                self.logger.info("롤백 완료")
                return {
                    "status": "success",
                    "restored_path": original_path,
                    "backup_used": backup_path
                }
        
        raise Exception("Invalid backup structure")
