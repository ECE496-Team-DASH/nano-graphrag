import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken


from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
    gemini_2_5_flash_complete,
    gemini_1_5_pro_complete,
    gemini_embedding,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    extract_entities_genkg,
    generate_community_report,
    get_chunks,
    local_query,
    global_query,
    naive_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: gemini_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    using_gemini: bool = True
    best_model_func: callable = gemini_2_5_flash_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gemini_2_5_flash_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # GenKG integration
    use_genkg_extraction: bool = False
    genkg_node_limit: int = 25
    genkg_llm_provider: str = "gemini"  
    genkg_model_name: str = "gemini-2.5-flash"
    genkg_create_visualization: bool = False
    genkg_output_path: Optional[str] = None

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )
            
        # Override with Gemini if using_gemini is True
        if self.using_gemini:
            logger.info("Using Gemini for LLM and embeddings")
            self.best_model_func = gemini_2_5_flash_complete
            self.cheap_model_func = gemini_2_5_flash_complete  
            self.embedding_func = gemini_embedding

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        # Configure GenKG if enabled
        if self.use_genkg_extraction:
            logger.info("Using GenKG for entity extraction")
            self.entity_extraction_func = extract_entities_genkg
            
            # Set default output path if not provided and visualization is enabled
            if self.genkg_create_visualization and not self.genkg_output_path:
                self.genkg_output_path = os.path.join(self.working_dir, "output.html")
                
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- generate GenKG visualizations if enabled
            if self.use_genkg_extraction and self.genkg_create_visualization:
                await self._generate_genkg_visualizations(inserting_chunks, new_docs)

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _generate_genkg_visualizations(self, inserting_chunks, new_docs):
        """
        Generate GenKG HTML and JSON visualizations after successful entity extraction.
        """
        try:
            logger.info("[GenKG Visualization] Generating output files...")
            
            # Import genkg 
            import sys
            # Try multiple possible genkg locations
            possible_genkg_paths = [
                os.path.join(self.working_dir, "..", "nano-graphrag", "genkg.py"),
                os.path.join(self.working_dir, "..", "..", "nano-graphrag", "genkg.py"),
                os.path.join(os.path.dirname(__file__), "..", "genkg.py"),
            ]
            
            genkg_found = False
            for genkg_path in possible_genkg_paths:
                if os.path.exists(genkg_path):
                    genkg_dir = os.path.dirname(genkg_path)
                    if genkg_dir not in sys.path:
                        sys.path.insert(0, genkg_dir)
                    genkg_found = True
                    break
                    
            if not genkg_found:
                raise ImportError(f"GenKG not found in any of: {possible_genkg_paths}")
                
            from genkg import GenerateKG
            
            # Initialize GenKG
            genkg = GenerateKG(
                llm_provider=self.genkg_llm_provider,
                model_name=self.genkg_model_name
            )
            
            # Generate visualizations using the chunks
            result = genkg.generate_knowledge_graph_from_chunks(
                chunks_dict=inserting_chunks,
                nodes_per_document=self.genkg_node_limit,
                output_path=self.genkg_output_path,
                create_visualization=True
            )
            
            # Also create the .dashkg.json file
            if self.genkg_output_path and result.get("knowledge_graph"):
                dashkg_path = self.genkg_output_path.replace('.html', '.dashkg.json')
                genkg.export_graph_to_dashkg_json(result["knowledge_graph"], dashkg_path)
                
            logger.info(f"[GenKG Visualization] Files generated: {self.genkg_output_path}")
            
        except Exception as e:
            logger.error(f"Error generating GenKG visualizations: {e}")
            # Don't fail the entire insertion if visualization fails
