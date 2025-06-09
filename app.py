import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime, timedelta
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from collections import deque
import time
import json
from typing import List, Dict, Tuple, Optional
import hashlib
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import concurrent.futures
from threading import Lock
from datetime import timezone
import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Add a lock for thread-safe operations
scrape_lock = Lock()

# Load environment variables
load_dotenv()

# Configuration
user_agent = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; MyScraper/1.0)")
headers = {"User-Agent": user_agent}
BBC_URL = "https://www.bbc.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

MAX_CHAT_HISTORY = 10
MAX_ARTICLES_PER_SECTION = 100
SCRAPE_CACHE_TIME = timedelta(hours=6)
DATA_DIR = "bbc_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=MAX_CHAT_HISTORY)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_scrape_time" not in st.session_state:
    st.session_state.last_scrape_time = None
if "scraped_articles" not in st.session_state:
    st.session_state.scraped_articles = []
if "user_interests" not in st.session_state:
    st.session_state.user_interests = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Utility functions
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text

def is_valid_url(url: str) -> bool:
    """Check if URL is valid and belongs to BBC domain."""
    parsed = urlparse(url)
    return (parsed.scheme in ('http', 'https') and 
            parsed.netloc.endswith('bbc.com') and
            not any(exclude in url.lower() for exclude in ['/live/', '/av/']))

def get_page_content(url: str) -> Tuple[str, Optional[datetime]]:
    """Fetch and parse page content with enhanced error handling."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return "", None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'aside', 
                           'header', 'form', 'button', 'svg', 'figure', 'ul', 'ol']):
            element.decompose()
            
        title = soup.title.string if soup.title else ""
        published_time = None
        
        # Extract publication time
        time_tag = soup.find('time', {'datetime': True})
        if time_tag:
            try:
                published_time = datetime.fromisoformat(time_tag['datetime'])
            except:
                pass
                
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.body
        if not main_content:
            return "", published_time
            
        # Extract content from BBC-specific containers
        for container in main_content.find_all(['div', 'section'], class_=re.compile(r'(story|article)-body')):
            text = container.get_text(separator=' ', strip=True)
            if len(text.split()) > 50:
                return clean_text(f"{title}\n\n{text}"), published_time
                
        # Fallback to all text
        text = main_content.get_text(separator=' ', strip=True)
        if len(text.split()) > 50:
            return clean_text(f"{title}\n\n{text}"), published_time
            
        return "", published_time
         
        if not published_time:
            published_time = datetime.now(timezone.utc)
        elif published_time.tzinfo is None:
            published_time = published_time.replace(tzinfo=timezone.utc)

                
    except Exception as e:
        print(f"Error fetching {url}: {str(e)[:100]}")
        return "", None

def get_nav_links(url: str) -> List[str]:
    """Extract section links from BBC News."""
    known_sections = [
        "https://www.bbc.com/news",
        "https://www.bbc.com/sport",
        "https://www.bbc.com/weather",
        "https://www.bbc.com/iplayer",
        "https://www.bbc.com/sounds",
        "https://www.bbc.com/bitesize",
        "https://www.bbc.com/cbeebies",
        "https://www.bbc.com/cbbc",
        "https://www.bbc.com/food",
        "https://www.bbc.com/business",
        "https://www.bbc.com/innovation",
        "https://www.bbc.com/culture",
        "https://www.bbc.com/future-planet",
        "https://www.bbc.com/audio",
        "https://www.bbc.com/video",
        "https://www.bbc.com/arts",
        "https://www.bbc.com/travel",
        "https://www.bbc.com/live",
        "https://www.bbc.com/news/world",
        "https://www.bbc.com/news/technology",
        "https://www.bbc.com/news/science_and_environment",
        "https://www.bbc.com/news/business",
        "https://www.bbc.com/news/health",
        "https://www.bbc.com/news/entertainment_and_arts"
    ]
    return [section for section in known_sections if is_valid_url(section)]

def scrape_articles_from_section(section_url: str, max_articles: int = MAX_ARTICLES_PER_SECTION) -> List[Dict]:
    """Enhanced article scraping with better URL discovery and date handling."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(section_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        seen_urls = set()
        
        # BBC article link patterns
        link_selectors = [
            ('a', {'href': re.compile(r'/news/[\w-]+-\d+$')}),
            ('a', {'href': re.compile(r'/sport/[\w-]+-\d+$')}),
            ('div[data-entityid="container-top-stories"] a', None),
            ('a[class*="block-link"]', None),
            ('a[class*="promo-link"]', None),
            ('a[data-testid="internal-link"]', None),
            ('a[class*="gs-c-promo-heading"]', None),
        ]
        
        for tag, attrs in link_selectors:
            if len(articles) >= max_articles:
                break
                
            if attrs:
                for link in soup.find_all(tag, attrs=attrs):
                    if len(articles) >= max_articles:
                        break
                        
                    href = link.get('href', '')
                    if not href or href.startswith('#') or 'live' in href.lower():
                        continue
                        
                    full_url = urljoin(BBC_URL, href)
                    if full_url in seen_urls or not is_valid_url(full_url):
                        continue
                        
                    article_content, published_time = get_page_content(full_url)
                    if article_content and len(article_content.split()) > 50:
                        if not published_time:
                            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', full_url)
                            if date_match:
                                year, month, day = map(int, date_match.groups())
                                published_time = datetime(year, month, day)
                        if not published_time:
                            published_time = datetime.now()
                            
                        title = link.get_text(strip=True) or "Untitled Article"
                        if len(title) < 3:
                            title = ' '.join(article_content.split()[:5]) + "..."
                            
                        articles.append({
                            'url': full_url,
                            'title': title,
                            'content': article_content,
                            'timestamp': published_time.isoformat(),
                            'section': section_url,
                            'scrape_time': datetime.now().isoformat()
                        })
                        seen_urls.add(full_url)
            else:
                for link in soup.select(tag):
                    if len(articles) >= max_articles:
                        break
                        
                    if link.has_attr('href'):
                        href = link['href']
                        if not href or href.startswith('#') or 'live' in href.lower():
                            continue
                            
                        full_url = urljoin(BBC_URL, href)
                        if full_url in seen_urls or not is_valid_url(full_url):
                            continue
                            
                        article_content, published_time = get_page_content(full_url)
                        if article_content and len(article_content.split()) > 50:
                            if not published_time:
                                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', full_url)
                                if date_match:
                                    year, month, day = map(int, date_match.groups())
                                    published_time = datetime(year, month, day)
                            if not published_time:
                                published_time = datetime.now()
                                
                            title = link.get_text(strip=True) or "Untitled Article"
                            if len(title) < 3:
                                title = ' '.join(article_content.split()[:5]) + "..."
                                
                            articles.append({
                                'url': full_url,
                                'title': title,
                                'content': article_content,
                                'timestamp': published_time.isoformat(),
                                'section': section_url,
                                'scrape_time': datetime.now().isoformat()
                            })
                            seen_urls.add(full_url)
                            
        articles.sort(key=lambda x: x['timestamp'], reverse=True)
        return articles[:max_articles]  
        
    except Exception as e:
        print(f"Error scraping section {section_url}: {e}")
        return []
    
def setup_scheduler():
    """Initialize the background scheduler"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=scrape_bbc_news_background,
        trigger='interval',
        hours=1,  # Auto-scrape every hour
        next_run_time=datetime.now()  # Run immediately on startup
    )
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

def safe_datetime_diff(dt1, dt2):
    """Safe datetime comparison that handles both naive and aware datetimes"""
    if dt1.tzinfo is None or dt2.tzinfo is None:
        # If either datetime is naive, make both naive for comparison
        dt1 = dt1.replace(tzinfo=None)
        dt2 = dt2.replace(tzinfo=None)
    return dt1 - dt2    

def scrape_bbc_news_background():
    """Optimized background scraping with thread safety and incremental updates"""
    try:
        if st.session_state.get('scraping_in_progress', False):
            return
            
        with scrape_lock:
            st.session_state.scraping_in_progress = True
            print("Starting optimized background scrape...")
            
            # Load existing data with timezone handling
            existing_data = load_scraped_articles() or []
            existing_urls = {a['url'] for a in existing_data}
            
            # Determine sections to scrape
            sections_to_scrape = (
                st.session_state.user_interest_urls 
                if st.session_state.get('user_interest_urls') 
                else get_nav_links(BBC_URL)
            )
            
            # Thread-safe scraping
            new_articles = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for url in sections_to_scrape:
                    # Only scrape sections that haven't been checked recently
                    last_scraped = get_last_scraped_time_for_section(url, existing_data)
                    if (not last_scraped or 
                        safe_datetime_diff(datetime.now(timezone.utc), last_scraped) > timedelta(hours=1)):
                        futures.append(executor.submit(scrape_articles_from_section, url))
                
                for future in concurrent.futures.as_completed(futures):
                    articles = future.result()
                    new_articles.extend(a for a in articles if a['url'] not in existing_urls)
                    time.sleep(1)  # Conservative delay
            
            if new_articles:
                # Process new articles with timezone info
                for article in new_articles:
                    if 'timestamp' not in article:
                        article['timestamp'] = datetime.now(timezone.utc)
                    elif isinstance(article['timestamp'], str):
                        dt = datetime.fromisoformat(article['timestamp'])
                        article['timestamp'] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                
                updated_data = new_articles + existing_data
                updated_data.sort(
                    key=lambda x: x['timestamp'], 
                    reverse=True
                )[:MAX_ARTICLES_PER_SECTION*10]
                
                # Save with proper serialization
                with open(os.path.join(DATA_DIR, "scraped_articles.json"), 'w') as f:
                    json.dump(updated_data, f, default=str)
                
                print(f"Added {len(new_articles)} new articles")
                
                # Safe vectorstore update
                if 'vectorstore' in st.session_state:
                    try:
                        create_vector_store(updated_data)
                        print("Vector store updated")
                    except Exception as e:
                        print(f"Vectorstore update error: {e}")
            
            st.session_state.last_scrape_time = datetime.now(timezone.utc).isoformat()
            
    except Exception as e:
        print(f"Background scrape error: {str(e)}")
    finally:
        st.session_state.scraping_in_progress = False

def get_last_scraped_time_for_section(section_url, existing_data):
    """Get the last time a section was scraped from existing data"""
    section_articles = [a for a in existing_data if a.get('section') == section_url]
    if not section_articles:
        return None
    
    try:
        latest = max(
            datetime.fromisoformat(a['scrape_time']) if isinstance(a['scrape_time'], str)
            else a['scrape_time']
            for a in section_articles
        )
        return latest.replace(tzinfo=timezone.utc) if latest.tzinfo is None else latest
    except:
        return None        

def scrape_bbc_news(force=False) -> Optional[List[Dict]]:
    """Smart scraping function that checks last scrape time and only gets new content"""
    # Check if we need to scrape
    last_scrape = None
    if st.session_state.last_scrape_time:
        try:
            last_scrape = datetime.fromisoformat(st.session_state.last_scrape_time)
            if last_scrape.tzinfo is None:
                last_scrape = last_scrape.replace(tzinfo=timezone.utc)
        except:
            last_scrape = None
    
    needs_scrape = force or not last_scrape or safe_datetime_diff(datetime.now(timezone.utc), last_scrape) > SCRAPE_CACHE_TIME
    
    if not needs_scrape:
        print("Using cached data (recently scraped)")
        return st.session_state.scraped_articles
        
    # Show scraping UI only if forced or first run
    if force or not st.session_state.get('initialized'):
        with st.spinner("Fetching latest news..."):
            return _perform_scraping_ui()
    else:
        # Return cached data while background scrape runs
        return st.session_state.scraped_articles

def _perform_scraping_ui():
    """UI-visible scraping with progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    existing_data = load_scraped_articles() or []
    existing_urls = {a['url'] for a in existing_data}
    new_articles = []
    
    sections = get_nav_links(BBC_URL)
    for i, section in enumerate(sections):
        progress = int((i / len(sections)) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Checking {section}...")
        
        articles = scrape_articles_from_section(section)
        new_articles.extend(a for a in articles if a['url'] not in existing_urls)
        time.sleep(1)  # Respectful delay
    
    if new_articles:
        updated_data = new_articles + existing_data
        updated_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        with open(os.path.join(DATA_DIR, "scraped_articles.json"), 'w') as f:
            json.dump(updated_data[:MAX_ARTICLES_PER_SECTION*10], f)
        
        st.success(f"Added {len(new_articles)} new articles")
        st.session_state.vectorstore = create_vector_store(updated_data)
    else:
        st.success("No new articles found")
    
    st.session_state.last_scrape_time = datetime.now().isoformat()
    st.session_state.scraped_articles = updated_data[:MAX_ARTICLES_PER_SECTION*10]
    return st.session_state.scraped_articles

def load_scraped_articles() -> Optional[List[Dict]]:
    """Load previously scraped articles from file."""
    scraped_data_path = os.path.join(DATA_DIR, "scraped_articles.json")
    if os.path.exists(scraped_data_path):
        try:
            with open(scraped_data_path, 'r') as f:
                articles = json.load(f)
                for article in articles:
                    article['timestamp'] = datetime.fromisoformat(article['timestamp'])
                return articles
        except Exception as e:
            print(f"Error loading scraped articles: {e}")
    return None

def needs_rescrape() -> bool:
    """Check if we need to re-scrape based on last scrape time."""
    if not st.session_state.last_scrape_time:
        return True
        
    try:
        last_scrape = datetime.fromisoformat(st.session_state.last_scrape_time)
        return datetime.now() - last_scrape > SCRAPE_CACHE_TIME
    except:
        return True

def create_vector_store(articles: List[Dict]) -> FAISS:
    """Create FAISS vector store from articles with enhanced metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    docs = []
    for article in articles:
        doc = Document(
            page_content=article['content'],
            metadata={
                "source": article['url'],
                "title": article['title'],
                "timestamp": article['timestamp'].isoformat() if isinstance(article['timestamp'], datetime) else article['timestamp'],
                "section": article['section'],
                "scrape_time": article.get('scrape_time', datetime.now().isoformat())
            }
        )
        docs.append(doc)
        
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(os.path.join(DATA_DIR, "bbc_news_faiss_index"))
    return vectorstore

def get_latest_news_by_topics(topics: List[str], vectorstore: FAISS, num_articles: int = 5) -> str:
    """Get latest news for specific topics with detailed information."""
    results = []
    
    for topic in topics:
        # Get relevant documents for the topic, sorted by date
        docs = vectorstore.similarity_search(topic, k=num_articles*2)
        docs.sort(key=lambda x: x.metadata.get("timestamp", ""), reverse=True)
        docs = docs[:num_articles]

        print(docs)
        
        if not docs:
            results.append(f"## {topic.capitalize()}\nNo recent articles found for {topic}.")
            continue
            
        topic_result = [f"## {topic.capitalize()} - Latest News"]
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Untitled Article")
            url = doc.metadata.get("source", "#")
            timestamp = doc.metadata.get("timestamp", "")
            
            try:
                date_str = datetime.fromisoformat(timestamp).strftime("%B %d, %Y")
            except:
                date_str = timestamp
                
            summary = " ".join(doc.page_content.split()[:50]) + "..."
            
            topic_result.append(
                f"{i}. **{title}**\n"
                f"   - *Published*: {date_str}\n"
                f"   - *Summary*: {summary}\n"
                f"   - [Read more]({url})"
            )
            
        results.append("\n\n".join(topic_result))
        
    return "\n\n".join(results)

def scrape_bbc_headlines(section_url: str, max_results: int = 5) -> List[Dict]:
    """Scrape BBC headlines with enhanced selectors"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(section_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        headlines = []
        selectors = [
            'a[class*="promo-heading"] h3',    # Main promo headlines
            'a[data-testid="internal-link"] h2', # Internal links
            'div[class*="promo-text"] h3',      # Promo text
            'h3[class*="title"]',               # Generic titles
            'li[class*="lx-stream-post"] h3',   # Stream posts
            'a[class*="gs-c-promo-heading"] h3' # Global promo
            'a[data-testid="internal-link"]',  # Primary modern BBC selector
            'a[class*="ssrcss-"]',             # New BBC class pattern
            'div[data-component="card"] a',    # Card component links
            'a[href*="/sport/"]',              # Any sport links
            'h3 a',                            # Simple h3 links fallback

        ]
        
        for selector in selectors:
            for headline in soup.select(selector):
                if len(headlines) >= max_results:
                    break
                
                title = headline.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                link = headline.find_parent('a')
                if not link or not link.get('href'):
                    continue
                
                full_url = urljoin("https://www.bbc.com", link['href'])
                if not full_url.startswith('https://www.bbc.com'):
                    continue
                
                headlines.append({
                    'title': title,
                    'url': full_url,
                    'section': section_url
                })
        
        return headlines[:max_results]
    except Exception as e:
        print(f"Error scraping {section_url}: {e}")
        return []
    
def format_document(doc):
    """Helper to standardize document formatting"""
    timestamp = doc.metadata.get("timestamp", "")
    date_str = datetime.fromisoformat(timestamp).strftime("%b %d") if timestamp else "Recent"
    return (
        f"📰 {doc.metadata.get('title', 'BBC Article')}\n"
        f"📅 {date_str}\n"
        f"🔗 {doc.metadata.get('source', '#')}\n"
        f"Content:\n{doc.page_content[:500]}..."
    )

def get_response_with_sources(user_query: str, vectorstore: FAISS) -> Tuple[str, List[str]]:
    """Enhanced unified function with smarter prompts for all query types"""
    query_lower = user_query.lower()
    current_date = datetime.now().strftime("%B %d, %Y")  
    # Complete topic to URL mapping
    TOPIC_TO_URL = {
        'sports': 'https://www.bbc.com/sport',
        'art': 'https://www.bbc.com/arts',
        'culture': 'https://www.bbc.com/culture',
        'business': 'https://www.bbc.com/business',
        'news': 'https://www.bbc.com/news',
        'technology': 'https://www.bbc.com/news/technology',
        'science': 'https://www.bbc.com/news/science_and_environment',
        'health': 'https://www.bbc.com/news/health',
        'entertainment': 'https://www.bbc.com/news/entertainment_and_arts',
        'weather': 'https://www.bbc.com/weather',
        'travel': 'https://www.bbc.com/travel',
        'politics': 'https://www.bbc.com/news/politics',
        'world': 'https://www.bbc.com/news/world',
        'live': 'https://www.bbc.com/news/live',
        'football': 'https://www.bbc.com/sport/football',
        'tennis': 'https://www.bbc.com/sport/tennis',
        'finance': 'https://www.bbc.com/news/business'
    }      
  
    # 1. First handle "my interested topics" queries with special processing
    should_use_interests = (
        any(phrase in query_lower for phrase in [
            "my interested topic", 
            "my interested topics", 
            "my interests",
            "based on my interests",
            "according to my preferences"
        ]) or 
        (
            any(q_type in query_lower for q_type in [
                "summary", 
                "overview", 
                "latest", 
                "recent", 
                "what's happening", 
                "current situation",
                "what's new",
                "updates",
                "news"
            ]) and 
            st.session_state.user_interests
        )
    )

    if should_use_interests:
        if not st.session_state.user_interests:
            return "Please set your interests in the sidebar configuration.", []
        
        # Prepare context from all interests
        all_docs = []
        for interest in st.session_state.user_interests:
            docs = vectorstore.similarity_search(interest, k=5)
            all_docs.extend(docs)
        
        # Remove duplicates and prepare context
        unique_docs = {}
        source_urls = set()
        context = []
        
        for doc in all_docs:
            url = doc.metadata.get("source")
            if url not in unique_docs:
                unique_docs[url] = doc
                timestamp = doc.metadata.get("timestamp", "")
                date_str = datetime.fromisoformat(timestamp).strftime("%B %d") if timestamp else "recently"
                context.append(
                    f"📰 {doc.metadata.get('title', 'BBC Article')}\n"
                    f"📅 {date_str}\n"
                    f"🔗 {url}\n"
                    f"Content:\n{doc.page_content[:800]}{'...' if len(doc.page_content) > 800 else ''}"
                )
                source_urls.add(url)
                
                # Determine response type based on query keywords
                if "summary" in query_lower or "overview" in query_lower:
                    prompt_template = """Create a comprehensive summary about these topics: {user_interests}
                    Articles: {context}
                    
                    Required Format:
                    1. **Executive Summary** (1-2 paragraphs)
                    2. **Key Points**:
                    - 3-5 bullet points of most important developments
                    - Include relevant dates and sources
                    3. **Analysis**: 
                    - 1 paragraph connecting the developments
                    - Highlight any trends or patterns"""
                    
                elif "latest news" in query_lower or "recent update" in query_lower:
                    prompt_template = """List the latest news about: {user_interests}
                    Articles: {context}
                    
                    Format:
                    - For each topic: [Title](URL) • Date • 1-sentence update
                    - Sort by newest first
                    - Highlight today's news with '🆕'
                    - Group by interest category"""
                    
                elif "what's happening" in query_lower or "current situation" in query_lower:
                    prompt_template = """Explain current situation for: {user_interests}
                    Articles: {context}
                    
                    Structure as:
                    1. Current Status (paragraph)
                    2. Major Events (bullets)
                    3. Future Outlook (paragraph)"""
                    
                else:  # Default general response about interests
                    prompt_template = """Analyze these topics: {user_interests}
                    Articles: {context}
                    
                    Include:
                    - Key facts with dates
                    - Important developments
                    - Related connections between topics"""
                
                # Generate the response
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | llm | StrOutputParser()
                
                try:
                    response = chain.invoke({
                        "input": user_query,
                        "context": "\n\n---\n\n".join(context),
                        "current_date": current_date,
                        "user_interests": ", ".join(st.session_state.user_interests)
                    })
                    
                    if source_urls and "Sources:" not in response.lower():
                        response += f"\n\nSources:\n" + "\n".join(f"- [{url.split('/')[-1]}]({url})" for url in list(source_urls)[:3])
                    return response, list(source_urls)
                    
                except Exception as e:
                    return f"Error processing your request: {str(e)[:100]}", []
               
            # 1. Handle system questions
    system_phrases = {
        'who are you': "I'm your BBC News Assistant, here to provide the latest news and information from BBC. "
                      "I can fetch headlines, answer questions about current events, and keep you updated on topics you care about.",
        'what can you do': "I can:\n"
                          "- Fetch latest BBC headlines on any topic\n"
                          "- Answer questions about current news\n"
                          "- Provide detailed information from BBC articles\n"
                          "- Track your interests and show relevant news\n\n"
                          "Try asking about sports, politics, technology or any current affairs!",
        'your purpose': "My purpose is to make BBC News more accessible and helpful for you. "
                       "I can quickly find information so you don't have to browse through multiple articles.",
        'help': "Here's how I can help:\n\n"
                "• Ask for latest news: 'What's happening in tech?'\n"
                "• Set your interests: 'I like sports and politics'\n"
                "• General questions: 'Tell me about the UK election'\n"
                "• Specific queries: 'What did BBC say about climate change?",
        'hello': "Hello! I'm your BBC News assistant. How can I help you today?",
        'hi': "Hi there! Ready to explore the latest BBC News with me?"
    }
    
    for phrase, response in system_phrases.items():
        if re.search(r'\b' + re.escape(phrase) + r'\b', query_lower):
            return response, []
        
                    # 2. Query type detection
            
    interest_keywords = ['interest', 'interested', 'prefer', 'favorite', 'topic', 'i like', 'show me']
    is_interest_query = any(keyword in query_lower for keyword in interest_keywords)
    is_latest_query = any(word in query_lower for word in ['latest', 'recent', 'newest', 'current', 'today', 'breaking'])
    is_summary_query = any(word in query_lower for word in ['summary', 'summarize', 'overview', 'brief'])

    # 3. Handle interest queries first (your original logic)
    if is_interest_query:
        mentioned_topics = []
        for topic in TOPIC_TO_URL:
            if topic in query_lower:
                mentioned_topics.append(topic)
        
        topics_to_check = mentioned_topics or [t.lower() for t in st.session_state.get('user_interests', [])]
        
        if not topics_to_check:
            return ("I'm your BBC News assistant. I can show you news about:\n\n"
                   "• Sports ⚽\n• Politics 🏛️\n• Technology 💻\n• Business 💰\n• Health 🏥\n\n"
                   "What would you like? Try: 'Show me sports news' or 'I like technology'"), []

        response_lines = []
        source_urls = []
        
        for topic in sorted(set(topics_to_check)):
            if topic in TOPIC_TO_URL:
                section_url = TOPIC_TO_URL[topic]
                
                if not is_valid_url(section_url):
                    continue
                    
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    response = requests.get(section_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    headlines = []
                    
                    selectors = [
                        'a[class*="promo-heading"] h3',
                        'a[data-testid="internal-link"] h2',
                        'div[class*="promo-text"] h3',
                        'h3[class*="title"]',
                        'li[class*="lx-stream-post"] h3',
                        'a[class*="gs-c-promo-heading"] h3',
                        'a[data-testid="internal-link"]',  # Primary modern BBC selector
                        'a[class*="ssrcss-"]',             # New BBC class pattern
                        'div[data-component="card"] a',    # Card component links
                        'a[href*="/sport/"]',              # Any sport links
                        'h3 a',                            # Simple h3 links fallback
                    ]
                    
                    for selector in selectors:
                        for headline in soup.select(selector):
                            if len(headlines) >= 5:
                                break
                            title = headline.get_text(strip=True)
                            if not title or len(title) < 10:
                                continue
                            link = headline.find_parent('a')
                            if not link or not link.get('href'):
                                continue
                            full_url = urljoin(BBC_URL, link['href'])
                            if not is_valid_url(full_url):
                                continue
                            headlines.append({
                                'title': title,
                                'url': full_url
                            })
                    
                    if headlines:
                        response_lines.append(f"\n📰 **Latest {topic.capitalize()} Headlines:**")
                        for i, headline in enumerate(headlines[:5], 1):
                            response_lines.append(f"{i}. {headline['title']} [Read more]({headline['url']})")
                            source_urls.append(headline['url'])
                    else:
                        response_lines.append(f"\n⚠️ Couldn't fetch current {topic} headlines. The section might be updating.")
                except Exception as e:
                    print(f"Error scraping {section_url}: {e}")
                    response_lines.append(f"\n⚠️ Could not retrieve current {topic} headlines due to an error.")
        
        if response_lines:
            return "\n".join(response_lines), source_urls
        return "I couldn't find recent articles on these topics. The BBC sections might be temporarily unavailable.", []

    
    # 4. Fetch documents based on query type
    if is_latest_query and st.session_state.user_interests:
        # Latest news for user interests
        docs = []
        for interest in st.session_state.user_interests:
            docs.extend(vectorstore.similarity_search(interest, k=3))
        docs = sorted(docs, key=lambda x: x.metadata.get("timestamp", ""), reverse=True)[:5]
    elif is_latest_query:
        # General latest news request
        docs = []
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)
            docs.append(doc)
        docs = sorted(docs, key=lambda x: x.metadata.get("timestamp", ""), reverse=True)[:5]
    else:
        # General or interest-based query
        docs = vectorstore.similarity_search(user_query, k=3)
    
    # Prepare context with dates and sources
    context = []
    source_urls = set()
    for doc in docs:
        timestamp = doc.metadata.get("timestamp", "")
        date_str = datetime.fromisoformat(timestamp).strftime("%B %d") if timestamp else "recently"
        source_url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "BBC News Article")
        
        context.append(
            f"📰 {title}\n"
            f"📅 {date_str}\n"
            f"🔗 {source_url}\n"
            f"Content:\n{doc.page_content[:800]}{'...' if len(doc.page_content) > 800 else ''}"
        )
        if source_url:
            source_urls.add(source_url)
    
    # 4. Handle general factual questions
    docs = vectorstore.similarity_search(user_query, k=3 if is_summary_query else 3)  # Could use more docs for summaries
    # Add this debug print to show retrieval sources
    print(f"\n🔍 Retrieved Documents for: '{user_query}'\n")
    for i, doc in enumerate(docs, 1):
        print(f"📄 Document {i}:")
        print(f"   Title: {doc.metadata.get('title', 'Untitled')}")
        print(f"   Source: {doc.metadata.get('source', 'Unknown URL')}")
        print(f"   Date: {doc.metadata.get('timestamp', 'Unknown date')}")
        print(f"   Content Preview: {doc.page_content[:200]}...\n")

    
    if not docs:
        return ("I couldn't find relevant information in recent BBC articles. "
               "I specialize in current news - try asking about recent events or "
               "say 'latest news' to see what's available."), []
    
    # Prepare context with dates
    context = []
    source_urls = set()
    for doc in docs:
        timestamp = doc.metadata.get("timestamp", "")
        date_str = datetime.fromisoformat(timestamp).strftime("%B %d") if timestamp else "recently"
        source_url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "BBC News Article")
        
        context.append(
            f"📰 {title}\n"
            f"📅 {date_str}\n"
            f"🔗 {source_url}\n"
            f"Content:\n{doc.page_content[:800]}{'...' if len(doc.page_content) > 800 else ''}"
        )
        if source_url:
            source_urls.add(source_url)

    if is_interest_query: 
            prompt_template = """

        You are a BBC News assistant. Provide the latest news based on:
        
        Current date: {current_date}
        User interests: {user_interests}
        
        Context:
        {context}
        
        Question: {input}
        
        Guidelines:
        1. List the 5 most recent {context} items
        2. Include publication dates
        3. Provide source URLs
        4. Be concise
        
        Answer:
        """
    elif is_latest_query:
        prompt_template = """You're a BBC News assistant providing the latest updates.
        Current Date: {current_date}
        
        Latest Articles:
        {context}
        
        Question: {input}
        
        Guidelines:
        1. List articles in reverse chronological order (newest first)
        2. For each article:
           - Title as markdown link: [Title](URL)
           - 1-2 sentence summary
           - Clear publication date (format: "Today" if same day, else "Month Day")
        3. Highlight breaking news with 🚨 emoji
        4. If no very recent articles, mention "Most recent available:"
        
        Example:
        ## Latest Updates
        1. [Title](URL) - Summary... (Today)
        2. [Title](URL) - Summary... (Jun 15)"""
    elif is_summary_query:
        prompt_template = """You're a BBC News summarizer creating overviews.
        Current Date: {current_date}

        Articles to summarize:
        {context}

        Guidelines:
        1. Start with "Here's a summary of recent developments:"
        2. Create a cohesive 1 paragraph narrative combining all sources
        3. Include:
        - Key events/facts
        - Important dates
        - Different perspectives if available
        4. Maintain neutral, professional tone
        5. Use paragraph breaks for readability
        6. End with:
        "For more details:"
        - [Source 1](URL)
        - [Source 2](URL)

        Example:
        Summary of Recent Events:
        The main developments as of {current_date} indicate...
        [Paragraph 1: Overview]
        [Paragraph 2: Key details]
        [Paragraph 3: Additional context]

        For more details:
        - [Full article 1](URL)
        - [Full article 2](URL)"""
    else:  # General questions
        prompt_template ="""You're a BBC News expert answering questions.
        Current Date: {current_date}
        
        Relevant Articles:
        {context}
        
        Question: {input}
        
        Guidelines:
        1. Provide a direct 3-4 line answer first
        2. Include:
           - Key facts
           - Important dates (especially if 'latest' mentioned)
           - Source attribution
        3. If 'latest' mentioned but no recent articles, say:
           "Most recent information available:"
        4. Format:
           [Concise Answer]
           [Relevant details]
           Sources: [URL1], [URL2]"""
        
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "input": user_query,
            "context": "\n\n---\n\n".join(context),
            "current_date": current_date,
            # Add this new parameter:
            "user_interests": ", ".join(st.session_state.user_interests) if st.session_state.user_interests else "Not specified"
        })
        

        if source_urls and "Sources:" not in response and "sources:" not in response.lower():
                    response += f"\n\nSources:\n" + "\n".join(f"- [{url.split('/')[-1]}]({url})" for url in list(source_urls)[:3])            
        return response, list(source_urls)
    
    except Exception as e:
        return f"I encountered an error processing your request. Please try again later. Error: {str(e)[:100]}", []

def initialize_system():
    """Initialize the system with proper datetime handling"""
    if not st.session_state.initialized:
        # Start background scheduler
        setup_scheduler()
        
        # Load existing data with timezone handling
        if not st.session_state.scraped_articles:
            loaded_articles = load_scraped_articles()
            if loaded_articles:
                # Ensure all timestamps are timezone-aware
                for article in loaded_articles:
                    if 'timestamp' in article:
                        if isinstance(article['timestamp'], str):
                            try:
                                dt = datetime.fromisoformat(article['timestamp'])
                                article['timestamp'] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                            except ValueError:
                                article['timestamp'] = datetime.now(timezone.utc)
                        elif isinstance(article['timestamp'], datetime) and article['timestamp'].tzinfo is None:
                            article['timestamp'] = article['timestamp'].replace(tzinfo=timezone.utc)
                
                st.session_state.scraped_articles = loaded_articles
                
                # Set last scrape time to most recent article time
                if loaded_articles:
                    try:
                        last_scrape = max(
                            art['timestamp'] if isinstance(art['timestamp'], datetime)
                            else datetime.fromisoformat(art['timestamp'])
                            for art in loaded_articles
                        )
                        st.session_state.last_scrape_time = last_scrape.isoformat()
                    except:
                        st.session_state.last_scrape_time = datetime.now(timezone.utc).isoformat()
        
        # Load vectorstore
        if "vectorstore" not in st.session_state:
            try:
                st.session_state.vectorstore = FAISS.load_local(
                    os.path.join(DATA_DIR, "bbc_news_faiss_index"), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Vectorstore load error: {e}")
                st.session_state.vectorstore = None
        
        st.session_state.initialized = True

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="BBC News Q&A System", layout="wide", page_icon="📰")
    st.title("📰 BBC News Q&A System")
    initialize_system()

    TOPIC_OPTIONS = {
    "News": "https://www.bbc.com/news",
    "Sports": "https://www.bbc.com/sport",
    "Weather": "https://www.bbc.com/weather",
    "iPlayer": "https://www.bbc.com/iplayer",
    "Sounds": "https://www.bbc.com/sounds",
    "Bitesize": "https://www.bbc.com/bitesize",
    "CBeebies": "https://www.bbc.com/cbeebies",
    "CBBC": "https://www.bbc.com/cbbc",
    "Food": "https://www.bbc.com/food",
    "Business": "https://www.bbc.com/business",
    "Innovation": "https://www.bbc.com/innovation",
    "Culture": "https://www.bbc.com/culture",
    "Future Planet": "https://www.bbc.com/future-planet",
    "Audio": "https://www.bbc.com/audio",
    "Video": "https://www.bbc.com/video",
    "Arts": "https://www.bbc.com/arts",
    "Travel": "https://www.bbc.com/travel",
    "Live": "https://www.bbc.com/live",
    "World News": "https://www.bbc.com/news/world",
    "Technology": "https://www.bbc.com/news/technology",
    "Science & Environment": "https://www.bbc.com/news/science_and_environment",
    "Business News": "https://www.bbc.com/news/business",
    "Health": "https://www.bbc.com/news/health",
    "Entertainment & Arts": "https://www.bbc.com/news/entertainment_and_arts"
}

    with st.sidebar:
        st.header("News Preferences")
        
        with st.expander("Select Your Interests"):
            # Multi-select dropdown for topics
            if 'user_interests' not in st.session_state:
                st.session_state.user_interests = []
            if 'user_interest_urls' not in st.session_state:
                st.session_state.user_interest_urls = []

            # Multi-select dropdown for topics - uses session state as default
            selected_topics = st.multiselect(
                "Choose news categories:",
                options=list(TOPIC_OPTIONS.keys()),
                default=st.session_state.user_interests,
                help="Select multiple categories that interest you"
            )
    
            
            if st.button("Update Preferences"):
                if selected_topics:
                    # Store both display names and URLs
                    st.session_state.user_interests = selected_topics
                    st.session_state.user_interest_urls = [TOPIC_OPTIONS[topic] for topic in selected_topics]
                    st.success(f"Preferences saved: {', '.join(selected_topics)}")
                else:
                    st.warning("Please select at least one interest")

                if st.button("Clear Preferences"):
                    st.session_state.user_interests = []
                    st.session_state.user_interest_urls = []
                    st.success("Preferences cleared!")    

                if st.session_state.user_interests:
                    st.markdown("---")
                    st.subheader("Current Preferences")
                    st.write(", ".join(st.session_state.user_interests))    

        if st.button("🔄 Scrape BBC News Now", help="Fetch fresh articles from BBC News"):
            with st.spinner("Scraping BBC News. This may take a few minutes..."):
                scraped_data = scrape_bbc_news()
                if scraped_data:
                    vectorstore = create_vector_store(scraped_data)
                    st.session_state.vectorstore = vectorstore
                    st.success("Vector store created and ready for questions!")

        st.markdown("---")
        st.subheader("System Status")

        if st.session_state.last_scrape_time:
            try:
                last_scrape = datetime.fromisoformat(st.session_state.last_scrape_time)
                st.write(f"📅 Last scraped: {last_scrape.strftime('%Y-%m-%d %H:%M')}")

                if st.session_state.scraped_articles:
                    st.metric("Stored Articles", len(st.session_state.scraped_articles))
                    newest = max(art['timestamp'] for art in st.session_state.scraped_articles)
                    oldest = min(art['timestamp'] for art in st.session_state.scraped_articles)
                    st.caption(f"Newest article: {newest.strftime('%Y-%m-%d')}")
                    st.caption(f"Oldest article: {oldest.strftime('%Y-%m-%d')}")
            except:
                st.write("📅 Last scraped: Unknown")

        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            st.success("✅ Vector store loaded")
            try:
                num_docs = len(st.session_state.vectorstore.index_to_docstore_id)
                st.caption(f"Indexed chunks: {num_docs}")
            except:
                pass
        else:
            st.warning("⚠️ No vector store available")

        if st.session_state.user_interests:
            st.markdown("---")
            st.subheader("Your Interests")
            st.write(", ".join(st.session_state.user_interests))

    # Main content area - modified layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Chat History")

        # Container for chat messages
        chat_container = st.container()

        # Display chat history in chronological order
        with chat_container:
            last_messages = list(st.session_state.chat_history)[-10:]  # This line added
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        if message.get("sources"):
                            with st.expander("View Sources"):
                                for source in message["sources"]:
                                    st.markdown(f"- [{source}]({source})")


        st.markdown(
            """
            <script>
                window.scrollTo(0, document.body.scrollHeight);
            </script>
            """,
            unsafe_allow_html=True
        )                           

        # Chat input at the bottom (fixed position)
        with st.container():
            user_query = st.chat_input("Type your question here...", key="chat_input")

            if user_query:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.now().isoformat()
                })

                # Process query and get response
                if "vectorstore" not in st.session_state or not st.session_state.vectorstore:
                    try:
                        st.session_state.vectorstore = FAISS.load_local(
                            os.path.join(DATA_DIR, "bbc_news_faiss_index"),
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                        st.rerun()
                    except:
                        st.error("Vector store not found. Trying to create one...")

                        if st.session_state.scraped_articles:
                            with st.spinner("Creating vector store..."):
                                st.session_state.vectorstore = create_vector_store(st.session_state.scraped_articles)
                                st.rerun()
                        else:
                            st.error("No scraped articles available. Please scrape BBC News first.")
                            st.stop()

                with st.spinner("Searching for answers..."):
                    try:
                        response, sources = get_response_with_sources(user_query, st.session_state.vectorstore)

                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat()
                        })

                        # Rerun to update the display
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)[:200]}")

    with col2:
        st.header("System Info")

        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            try:
                num_docs = len(st.session_state.vectorstore.index_to_docstore_id)
                st.metric("Indexed Chunks", num_docs)
            except:
                st.write("Index count unavailable")

        st.markdown("---")
        st.subheader("Recent Questions")
        for msg in list(st.session_state.chat_history)[-MAX_CHAT_HISTORY:]:
            if msg["role"] == "user":
                st.caption(f"❓ {msg['content'][:50]}...")
                
        st.markdown("---")
        st.subheader("Quick Actions")
        if st.button("Clear Chat History"):
            st.session_state.chat_history.clear()
            st.rerun()
            
        if st.button("Check Database"):
            if os.path.exists(os.path.join(DATA_DIR, "bbc_news_faiss_index")):
                st.success("Database files exist")
            else:
                st.warning("Database files not found")
                
        if st.button("Example Queries"):
            examples = [
                "What's the latest on Pakistan-India relations?",
                "Show me headlines about sports",
                "My interests are business and technology",
                "What are today's top news stories?",
                "Who are you and what can you do?"
            ]
            st.write("Try these:")
            for example in examples:
                st.code(example)

if __name__ == "__main__":
    main()
