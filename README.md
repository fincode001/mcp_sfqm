# MCP 서버 기능 품질 측정 시스템 (MCP-SFQM)

## 개요

MCP 서버 기능 품질 측정 시스템(MCP-SFQM)은 MCP 서버 풀의 세 가지 주요 문제를 해결하기 위해 설계되었습니다:

1. **서버 간 기능 중복**: 수백 개의 MCP 서버 간 중복 기능 식별 및 통합/분리 계획 수립
2. **기능적 충분성 측정**: 각 서버가 전문 MCP 서버로서 충분한 기능을 갖추었는지 평가
3. **품질 검증 및 오류 예방**: 통합/분리 과정에서 발생할 수 있는 오류 예방 및 품질 보장

## 주요 기능

- **서버 분석**: MCP 서버 파일의 구조, 기능, 품질, 복잡도 분석
- **기능 중복 탐지**: 서버 간 유사도 계산 및 중복 기능 식별
- **통합 권장사항**: 통합/분리가 필요한 서버 그룹 식별 및 우선순위 설정
- **품질 평가**: A/B/C/폐기 등급 체계를 통한 서버 품질 평가
- **보고서 생성**: 종합적인 분석 결과 및 개선 방안 제시

## 시스템 구성

MCP-SFQM은 다음과 같은 핵심 모듈로 구성되어 있습니다:

1. **Base Analyzer (`base_analyzer.py`)**: 
   - 모든 분석기의 기본 클래스
   - 공통 분석 기능 및 유틸리티 제공
   - 설정 관리 및 로깅 기능

2. **Server Analyzer (`server_analyzer.py`)**: 
   - MCP 서버 파일 전체 분석
   - 서버 유효성 검증 및 전문화 등급 평가
   - 코드 복잡도, 품질 지표, 의존성 분석

3. **Function Analyzer (`function_analyzer.py`)**: 
   - 서버 내 개별 함수 분석
   - 함수 복잡도, 매개변수, 문서화 수준 평가
   - 함수 간 유사도 및 중복성 분석

4. **통합 서버 분석 시스템 (`mcp_server_consolidation.py`)**:
   - 전체 MCP 서버 풀 분석
   - 중복 서버 그룹화 및 통합 계획 수립
   - 종합 보고서 생성

## 사용 방법

### 기본 사용법

```python
from mcp_sfqm.analyzers.server_analyzer import ServerAnalyzer
from mcp_sfqm.core.config import ConfigManager

async def analyze_mcp_server():
    # 설정 로드
    config = ConfigManager.get_instance()
    
    # 서버 분석기 초기화
    analyzer = ServerAnalyzer(config)
    
    # 서버 분석 실행
    server_path = 'path/to/your/mcp_server.py'
    result = await analyzer.analyze_server(server_path)
    
    # 결과 출력
    print(f'분석 점수: {result.get("analysis_score", 0)}/100')
    print(f'MCP 서버 유효성: {result.get("is_valid_mcp_server", False)}')
    
    # 전문화 등급 평가
    grade_info = await analyzer.evaluate_specialization_grade(server_path, [], {})
    print(f'전문화 등급: {grade_info.get("grade", "C")}')
```

### 전체 서버 풀 분석

```python
from mcp_server_consolidation import MCPServerConsolidator

# 통합 분석기 초기화
consolidator = MCPServerConsolidator("mcp_servers_path")

# 서버 스캔
consolidator.scan_final_servers()

# 통합 기회 분석
consolidator.identify_consolidation_opportunities()

# 보고서 생성
report = consolidator.generate_report()
```

## 평가 기준

### 서버 전문화 등급

- **A급**: 점수 80점 이상, 높은 코드 품질과 MCP 표준 준수
- **B급**: 점수 60-79점, 양호한 품질, 일부 개선 필요
- **C급**: 점수 40-59점, 기본 기능은 동작하나 품질 개선 필요
- **폐기**: 점수 40점 미만, 유효하지 않은 서버 또는 심각한 품질 문제

### 품질 측정 지표

- **MCP 유효성**: MCP 관련 import, 서버 클래스, 도구 정의 등 검증
- **문법 정확성**: Python 문법 오류 및 일반적인 코드 문제 검사
- **코드 복잡도**: 순환 복잡도, 인지 복잡도, 중첩 깊이 등 측정
- **품질 지표**: 주석 비율, 문서화 수준, 유지보수성 지수 등 계산

## 구현 기술

- **Python AST 모듈**: 코드 구문 분석 및 복잡도 측정
- **정규 표현식**: 코드 패턴 식별 및 기능 추출
- **비동기 프로그래밍**: `async/await`를 활용한 효율적인 분석 처리
- **데이터 클래스**: 구조화된 분석 결과 표현

## 통합 및 분리 원칙

1. **단일 책임 원칙(SRP)**: 각 서버는 명확하고 잘 정의된 책임만을 가져야 함
2. **도메인 중심 그룹화**: 유사 기능을 도메인별로 그룹화
3. **응집도 증대, 결합도 감소**: 관련 기능은 함께, 서버 간 의존성은 최소화
4. **데이터 소유권 명확화**: 데이터 변경은 소유자 서버를 통해서만 이루어지도록 설계

## 확장 및 개선 방향

- **실시간 모니터링**: 서버 성능 및 품질 실시간 모니터링 시스템 통합
- **자동 리팩토링**: 중복 코드 자동 리팩토링 기능 추가
- **웹 기반 대시보드**: 분석 결과 시각화 및 관리 인터페이스 개발
- **CI/CD 통합**: 지속적 통합/배포 파이프라인에 품질 검증 단계 추가

## 참고 문헌

- [MCP 서버 아키텍처](../_documentation/architecture.md)
- [서버 맵 및 관계](../_documentation/server_map.md)
- [중앙 레지스트리 관리](../_management/central_registry.json)

## 기여자

- 원본 아이디어 및 설계: fincode001
- 구현 및 테스트: AI 서비스 개발팀

## 라이선스

이 프로젝트는 사내 전용으로, 외부 공개 및 사용이 제한됩니다.
