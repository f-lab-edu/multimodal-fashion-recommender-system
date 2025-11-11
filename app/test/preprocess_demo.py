# test/preprocess_demo.py

from app.preprocess.pipeline import QueryPipeline

pipe = QueryPipeline.from_yaml()  # slots/defaults.yaml 로드

q = "여름 린넨 반팔 셔츠 추천해줘. 연청 데님 반바지도 찾아줘!"
out = pipe.process(q, include_tags_for_embed=True)
print(out["normalized_query"])
print("-" * 40)
print(out["slots"])
print("-" * 40)
print(out["query_for_embed"])
print("-" * 40)
