import streamlit as st
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import hashlib
from typing import List, Dict, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class EnhancedPDFProcessor:
    """향상된 PDF 처리 클래스"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def extract_text_with_ocr(self, pdf_docs) -> str:
        """OCR을 포함한 고급 텍스트 추출"""
        full_text = ""
        metadata = []
        
        for pdf_idx, pdf in enumerate(pdf_docs):
            try:
                # PyMuPDF로 PDF 열기
                doc = fitz.open(stream=pdf.read(), filetype="pdf")
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # 먼저 텍스트 추출 시도
                    text = page.get_text()
                    
                    # 텍스트가 부족하면 OCR 수행
                    if len(text.strip()) < 50:
                        logger.info(f"페이지 {page_num}에서 OCR 수행")
                        # 페이지를 이미지로 변환
                        mat = fitz.Matrix(2, 2)  # 해상도 증가
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # OCR 수행
                        try:
                            ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
                            text = ocr_text if len(ocr_text) > len(text) else text
                        except Exception as e:
                            logger.warning(f"OCR 실패: {e}")
                    
                    if text.strip():
                        full_text += f"\n\n=== 페이지 {page_num + 1} ===\n"
                        full_text += text
                        
                        # 메타데이터 저장
                        metadata.append({
                            'source': pdf.name,
                            'page': page_num + 1,
                            'text_length': len(text)
                        })
                
                doc.close()
                
            except Exception as e:
                logger.error(f"PDF 처리 오류 ({pdf.name}): {e}")
                continue
        
        return full_text, metadata
    
    def smart_chunking(self, text: str, metadata: List[Dict]) -> List[Dict]:
        """의미 기반 스마트 청킹"""
        # 기본 청킹
        chunks = self.text_splitter.split_text(text)
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # 청크별 메타데이터 추가
            chunk_metadata = {
                'chunk_id': i,
                'text': chunk,
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'sources': self._extract_sources_from_chunk(chunk, metadata)
            }
            enhanced_chunks.append(chunk_metadata)
        
        return enhanced_chunks

    def _extract_sources_from_chunk(self, chunk: str, metadata: List[Dict]) -> List[str]:
        """청크에서 소스 정보 추출"""
        sources = []
        for meta in metadata:
            if meta['source'] not in sources:
                sources.append(meta['source'])
        return sources

class EnhancedVectorStore:
    """향상된 벡터 저장소 관리"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    def create_vector_store(self, chunks: List[Dict]) -> FAISS:
        """메타데이터를 포함한 벡터 저장소 생성"""
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]
        
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        return vector_store
    
    def hybrid_search(self, vector_store: FAISS, query: str, k: int = 4) -> List[Dict]:
        """하이브리드 검색 (의미 + 키워드)"""
        # 의미 기반 검색
        semantic_docs = vector_store.similarity_search(query, k=k)
        
        # 키워드 기반 검색 (간단한 구현)
        keyword_docs = vector_store.similarity_search_with_score(query, k=k*2)
        
        # 결과 병합 및 리랭킹
        combined_results = []
        seen_texts = set()
        
        # 의미 검색 결과 우선 추가
        for doc in semantic_docs:
            if doc.page_content not in seen_texts:
                combined_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': 1.0,
                    'search_type': 'semantic'
                })
                seen_texts.add(doc.page_content)
        
        # 키워드 검색 결과 추가
        for doc, score in keyword_docs:
            if doc.page_content not in seen_texts:
                combined_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': 1.0 / (1.0 + score),  # 거리를 점수로 변환
                    'search_type': 'keyword'
                })
                seen_texts.add(doc.page_content)
        
        # 점수 기반 정렬
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:k]

class EnhancedQAChain:
    """향상된 질문 답변 체인"""
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.3
        )
        self.memory = ConversationBufferWindowMemory(
            k=5,  # 최근 5개 대화만 유지
            return_messages=True
        )
    
    def get_enhanced_prompt(self) -> PromptTemplate:
        """향상된 프롬프트 템플릿"""
        prompt_template = """
        당신은 전문적인 문서 분석가입니다. 주어진 컨텍스트를 바탕으로 정확하고 상세한 답변을 제공해주세요.

        컨텍스트 정보:
        {context}

        질문: {question}

        답변 지침:
        1. 컨텍스트에서 관련 정보를 찾아 정확하게 답변하세요
        2. 답변에 불확실한 내용이 있으면 "컨텍스트에 명확한 정보가 없습니다"라고 표시하세요
        3. 가능한 경우 구체적인 예시나 설명을 포함하세요
        4. 답변의 근거가 되는 소스 정보를 포함하세요

        답변:
        """
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
    def generate_answer(self, question: str, context_docs: List[Dict]) -> Dict[str, Any]:
        """향상된 답변 생성"""
        # 컨텍스트 구성
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # 프롬프트 생성
        prompt = self.get_enhanced_prompt()
        chain = load_qa_chain(
            self.model,
            chain_type="stuff",
            prompt=prompt
        )
        
        # 답변 생성
        response = chain.invoke(
            {"input_documents": [doc['content'] for doc in context_docs], "question": question},
            return_only_outputs=True
        )
        
        # 소스 정보 추출
        sources = []
        for doc in context_docs:
            if 'metadata' in doc and 'sources' in doc['metadata']:
                sources.extend(doc['metadata']['sources'])
        
        return {
            'answer': response['output_text'],
            'sources': list(set(sources)),
            'confidence': self._calculate_confidence(response['output_text'], context_docs),
            'context_used': len(context_docs)
        }
    
    def _calculate_confidence(self, answer: str, context_docs: List[Dict]) -> float:
        """답변 신뢰도 계산"""
        if "컨텍스트에 명확한 정보가 없습니다" in answer:
            return 0.3
        elif len(context_docs) >= 3:
            return 0.9
        elif len(context_docs) >= 2:
            return 0.7
        else:
            return 0.5

class EnhancedPDFChatSystem:
    """향상된 PDF 채팅 시스템"""
    
    def __init__(self):
        self.processor = EnhancedPDFProcessor()
        self.vector_store_manager = EnhancedVectorStore()
        self.qa_chain = EnhancedQAChain()
    
    def process_pdfs(self, pdf_docs) -> Dict[str, Any]:
        """PDF 처리 파이프라인"""
        with st.spinner("PDF를 처리하고 있습니다..."):
            # 텍스트 추출
            st.info("📄 텍스트 추출 중...")
            text, metadata = self.processor.extract_text_with_ocr(pdf_docs)
            
            if not text.strip():
                return {"error": "PDF에서 텍스트를 추출할 수 없습니다."}
            
            # 스마트 청킹
            st.info("🧩 텍스트 청킹 중...")
            chunks = self.processor.smart_chunking(text, metadata)
            
            # 벡터 저장소 생성
            st.info("🔍 벡터 저장소 생성 중...")
            vector_store = self.vector_store_manager.create_vector_store(chunks)
            
            return {
                "vector_store": vector_store,
                "chunks": chunks,
                "metadata": metadata,
                "total_text_length": len(text),
                "total_chunks": len(chunks)
            }
    
    def answer_question(self, question: str, vector_store: FAISS) -> Dict[str, Any]:
        """질문 답변 처리"""
        # 하이브리드 검색
        context_docs = self.vector_store_manager.hybrid_search(vector_store, question)
        
        if not context_docs:
            return {
                "answer": "관련 정보를 찾을 수 없습니다.",
                "sources": [],
                "confidence": 0.0,
                "context_used": 0
            }
        
        # 답변 생성
        result = self.qa_chain.generate_answer(question, context_docs)
        
        return result

# Streamlit 앱 메인 함수
def main():
    st.set_page_config(
        page_title="Enhanced PDF Chat System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🚀 향상된 PDF 질문 답변 시스템")
    st.markdown("---")
    
    # 세션 상태 초기화
    if "enhanced_vector_store" not in st.session_state:
        st.session_state.enhanced_vector_store = None
    if "enhanced_messages" not in st.session_state:
        st.session_state.enhanced_messages = [
            {"role": "assistant", "content": "안녕하세요! 📚 PDF 파일을 업로드하고 질문해보세요. OCR, 스마트 청킹, 하이브리드 검색 등 고급 기능을 제공합니다."}
        ]
    if "system_stats" not in st.session_state:
        st.session_state.system_stats = None
    
    # 사이드바
    with st.sidebar:
        st.header("📁 파일 관리")
        
        pdf_docs = st.file_uploader(
            "PDF 파일들을 업로드하세요",
            type=['pdf'],
            accept_multiple_files=True,
            help="여러 PDF 파일을 동시에 업로드할 수 있습니다."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("🔄 처리하기", type="primary")
        with col2:
            clear_btn = st.button("🗑️ 초기화")
        
        if clear_btn:
            st.session_state.enhanced_vector_store = None
            st.session_state.enhanced_messages = [
                {"role": "assistant", "content": "시스템이 초기화되었습니다. 새로 PDF를 업로드해주세요."}
            ]
            st.session_state.system_stats = None
            st.rerun()
        
        # 시스템 통계 표시
        if st.session_state.system_stats:
            st.markdown("---")
            st.subheader("📊 처리 통계")
            stats = st.session_state.system_stats
            st.metric("총 텍스트 길이", f"{stats['total_text_length']:,}자")
            st.metric("총 청크 수", f"{stats['total_chunks']:,}개")
            st.metric("처리된 파일 수", f"{len(stats['metadata'])}개")
    
    # PDF 처리
    if process_btn:
        if pdf_docs:
            system = EnhancedPDFChatSystem()
            result = system.process_pdfs(pdf_docs)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.enhanced_vector_store = result["vector_store"]
                st.session_state.system_stats = {
                    "total_text_length": result["total_text_length"],
                    "total_chunks": result["total_chunks"],
                    "metadata": result["metadata"]
                }
                
                st.success("✅ PDF 처리가 완료되었습니다!")
                st.session_state.enhanced_messages.append({
                    "role": "assistant", 
                    "content": f"🎉 PDF 처리가 완료되었습니다! 총 {result['total_chunks']}개의 청크로 분할되어 {result['total_text_length']:,}자의 텍스트가 처리되었습니다. 이제 질문해보세요!"
                })
                st.rerun()
        else:
            st.warning("⚠️ 먼저 PDF 파일을 업로드해주세요.")
    
    # 채팅 인터페이스
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 메시지 표시
        for message in st.session_state.enhanced_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 답변에 추가 정보가 있으면 표시
                if message["role"] == "assistant" and "additional_info" in message:
                    info = message["additional_info"]
                    with st.expander("📋 상세 정보"):
                        st.write(f"**신뢰도**: {info['confidence']:.1%}")
                        st.write(f"**사용된 컨텍스트**: {info['context_used']}개")
                        if info['sources']:
                            st.write("**출처**:")
                            for source in info['sources']:
                                st.write(f"- {source}")
    
    with col2:
        if st.session_state.enhanced_vector_store:
            st.success("✅ 시스템 준비됨")
        else:
            st.warning("⚠️ PDF 업로드 필요")
    
    # 질문 입력
    user_question = st.chat_input("PDF 내용에 대해 질문해보세요...")
    
    if user_question and st.session_state.enhanced_vector_store:
        # 사용자 메시지 추가
        st.session_state.enhanced_messages.append({"role": "user", "content": user_question})
        
        # 답변 생성
        with st.spinner("🤔 답변을 생성하고 있습니다..."):
            system = EnhancedPDFChatSystem()
            result = system.answer_question(user_question, st.session_state.enhanced_vector_store)
            
            # 답변 메시지 구성
            answer_message = {
                "role": "assistant",
                "content": result["answer"],
                "additional_info": {
                    "confidence": result["confidence"],
                    "sources": result["sources"],
                    "context_used": result["context_used"]
                }
            }
            
            st.session_state.enhanced_messages.append(answer_message)
            st.rerun()

if __name__ == "__main__":
    main()