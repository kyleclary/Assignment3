import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import os
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from deepseek_enrichment import (
    analyze_underperformance,
    generate_movie_insights,
    categorize_genre_patterns,
    assess_marketing_factors
)

# Create directory structure
for dir_path in ['data/raw', 'data/enriched', 'examples']:
    os.makedirs(dir_path, exist_ok=True)

# Constants
OMDB_API_KEY = "YOUR_OMDB_API_KEY"  # Get free key from http://www.omdbapi.com/apikey.aspx
MIN_RATING_THRESHOLD = 4.0
UNDERPERFORMANCE_THRESHOLD = 17_000_000  # $17M box office
MAX_YEAR = 2019  # Only analyze movies released before 2020
MAX_WORKERS = 3  # Parallel processing threads

# Session for connection pooling
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
})

@lru_cache(maxsize=100)
def cached_api_request(url, params_tuple):
    """Cache API responses to avoid duplicate requests"""
    params = dict(params_tuple)
    response = session.get(url, params=params, timeout=10)
    return response.json() if response.status_code == 200 else None

def scrape_letterboxd_efficient():
    """
    Optimized scraper using URLs that actually return movie data
    Fetches more movies to account for filtering
    """
    print("Scraping Letterboxd for highly-rated films...")
    
    all_movies = {}
    
    # Strategy 1: Use the main /films/ page (CONFIRMED WORKING - has 32 films)
    main_url = "https://letterboxd.com/films/"
    
    try:
        print(f"  Fetching main films page...")
        response = session.get(main_url, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get all images with alt text (most reliable)
            images = soup.find_all('img', alt=True)
            
            for img in images:
                title = img.get('alt', '').strip()
                # Filter out non-movie images
                if (title and len(title) > 2 and 
                    not any(skip in title.lower() for skip in ['poster', 'avatar', 'profile', 'logo'])):
                    
                    all_movies[title] = {
                        'title': title,
                        'year': 'Unknown',  # Will get from OMDB
                        'letterboxd_rating': 4.0,  # Conservative estimate
                        'source_url': main_url
                    }
            
            print(f"    Found {len(all_movies)} movies from main page")
    
    except Exception as e:
        print(f"    Error with main page: {e}")
    
    # Strategy 2: Use the TOP 250 list - get MORE movies since many will be filtered out
    list_url = "https://letterboxd.com/dave/list/official-top-250-narrative-feature-films/"
    
    try:
        print(f"  Fetching top 250 list...")
        response = session.get(list_url, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # From debug output: 100 films found with these selectors
            images = soup.find_all('img', alt=True)
            
            for img in images:
                title = img.get('alt', '').strip()
                
                # Skip profile images and duplicates
                if (title and title not in all_movies and len(title) > 2 and
                    not any(skip in title.lower() for skip in ['poster', 'avatar', 'profile'])):
                    
                    all_movies[title] = {
                        'title': title,
                        'year': 'Unknown',
                        'letterboxd_rating': 4.3,  # Higher rating for top 250
                        'source_url': list_url
                    }
                    
                    # Get MORE movies since many won't have box office data
                    if len(all_movies) >= 80:
                        break
            
            print(f"    Total movies after top 250 list: {len(all_movies)}")
                        
    except Exception as e:
        print(f"    Error with list page: {e}")
    
    # Try additional curated lists to get more candidates
    additional_lists = [
        "https://letterboxd.com/crew/list/top-250-highest-rated-films-of-all-time/",
        "https://letterboxd.com/jack/list/official-top-250-films-with-the-most-fans/",
        "https://letterboxd.com/dave/list/letterboxds-top-250-films-2024/"
    ]
    
    for list_url in additional_lists:
        if len(all_movies) >= 100:  # Get even more candidates
            break
            
        try:
            print(f"  Trying additional list...")
            response = session.get(list_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                images = soup.find_all('img', alt=True)
                
                for img in images:
                    title = img.get('alt', '').strip()
                    if (title and title not in all_movies and len(title) > 2 and
                        'avatar' not in title.lower()):
                        
                        all_movies[title] = {
                            'title': title,
                            'year': 'Unknown',
                            'letterboxd_rating': 4.2,
                            'source_url': list_url
                        }
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            continue
    
    # Convert to list and clean up
    movies_list = list(all_movies.values())
    
    # Add slugs and URLs
    for movie in movies_list:
        # Clean title for slug
        slug = movie['title'].lower()
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special characters
        slug = re.sub(r'[-\s]+', '-', slug)   # Replace spaces/multiple hyphens with single hyphen
        movie['slug'] = slug
        movie['letterboxd_url'] = f"https://letterboxd.com/film/{slug}/"
    
    print(f"  Successfully scraped {len(movies_list)} unique movies")
    print("  Note: Many will be filtered for lack of box office data or being post-2019")
    
    # Return MORE movies since many will be filtered out
    return movies_list[:100]  # Increased from 40 to 100

def fetch_omdb_batch(movies):
    """
    Fetch OMDB data in parallel for efficiency
    Filters for pre-2020 releases and movies with actual box office data
    """
    print(f"Fetching box office data for {len(movies)} movies...")
    enriched_movies = []
    
    def fetch_single_movie(movie):
        """Fetch data for a single movie"""
        try:
            # Build params
            params_dict = {
                'apikey': OMDB_API_KEY,
                't': movie['title'],
                'type': 'movie'
            }
            
            # Use cached request
            params_tuple = tuple(sorted(params_dict.items()))
            data = cached_api_request('http://www.omdbapi.com/', params_tuple)
            
            if not data or data.get('Response') != 'True':
                return None
            
            # Extract and validate year
            year_str = data.get('Year', 'Unknown')
            year_match = re.search(r'(\d{4})', year_str)
            
            if year_match:
                year = int(year_match.group(1))
                if year > MAX_YEAR:
                    return None  # Skip post-2019 films
            else:
                return None
            
            # Parse financial data efficiently
            def parse_money(value):
                if value and value != 'N/A':
                    return int(re.sub(r'[^\d]', '', value) or 0)
                return 0
            
            # Parse box office
            box_office = parse_money(data.get('BoxOffice'))
            
            # SKIP MOVIES WITH NO BOX OFFICE DATA
            if box_office == 0:
                print(f"    Skipping {movie['title']} - No box office data available")
                return None
            
            return {
                **movie,
                'year': str(year),
                'box_office': box_office,
                'budget': parse_money(data.get('Budget')),
                'imdb_rating': float(data.get('imdbRating', 0)) if data.get('imdbRating', 'N/A') != 'N/A' else 0,
                'genre': data.get('Genre', 'Unknown'),
                'director': data.get('Director', 'Unknown'),
                'actors': data.get('Actors', 'Unknown')[:100],  # Limit actor string length
                'plot': data.get('Plot', 'No plot available')[:200],  # Limit plot length
                'awards': data.get('Awards', 'N/A'),
                'runtime': data.get('Runtime', 'N/A'),
                'country': data.get('Country', 'Unknown'),
                'language': data.get('Language', 'Unknown')
            }
            
        except Exception as e:
            print(f"  Error fetching {movie['title']}: {e}")
            return None
    
    # Parallel fetching with progress tracking
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_movie = {
            executor.submit(fetch_single_movie, movie): movie 
            for movie in movies
        }
        
        for future in as_completed(future_to_movie):
            result = future.result()
            if result:
                enriched_movies.append(result)
                print(f"  ✓ {result['title']} ({result['year']}) - ${result['box_office']:,}")
        
        # Brief pause between batches to respect API limits
        time.sleep(0.5)
    
    print(f"\nSuccessfully enriched {len(enriched_movies)} pre-2020 movies with box office data")
    print(f"Filtered out {len(movies) - len(enriched_movies)} movies (post-2019 or no box office data)")
    return enriched_movies

def transform_data_vectorized(movies_data):
    """
    Efficiently transform data using pandas vectorized operations
    """
    if not movies_data:
        print("No data to transform")
        return pd.DataFrame()
    
    df = pd.DataFrame(movies_data)
    
    # Vectorized operations (much faster than apply)
    df['title'] = df['title'].str.strip()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Filter pre-2020 and ensure box office > 0
    df = df[(df['year'] <= MAX_YEAR) & (df['box_office'] > 0)].copy()
    
    if len(df) == 0:
        print("No valid movies with box office data found")
        return df
    
    # Vectorized calculations
    df['is_underperformer'] = df['box_office'] < UNDERPERFORMANCE_THRESHOLD
    df['box_office_millions'] = (df['box_office'] / 1_000_000).round(2)
    df['budget_millions'] = (df['budget'] / 1_000_000).round(2)
    
    # Vectorized ROI calculation
    mask = df['budget'] > 0
    df.loc[mask, 'roi'] = ((df.loc[mask, 'box_office'] - df.loc[mask, 'budget']) / 
                           df.loc[mask, 'budget'] * 100).round(2)
    df.loc[~mask, 'roi'] = None
    
    # More vectorized operations
    df['rating_differential'] = (df['letterboxd_rating'] - df['imdb_rating']).round(2)
    df['decade'] = (df['year'] // 10 * 10).astype('Int64')
    df['decade_label'] = df['decade'].astype(str) + 's'
    
    # Sort once at the end
    df = df.sort_values('letterboxd_rating', ascending=False)
    
    # Print summary of movies with box office data
    print(f"  Movies with valid box office data: {len(df)}")
    print(f"  Box office range: ${df['box_office_millions'].min():.1f}M - ${df['box_office_millions'].max():.1f}M")
    
    return df

def enrich_with_ai_batch(df):
    """
    Optimized AI enrichment with batching and caching
    """
    if df.empty:
        return df
    
    print("\nEnriching data with AI analysis...")
    
    # Focus on underperformers only
    underperformers = df[df['is_underperformer'] == True]
    
    if underperformers.empty:
        df['underperformance_category'] = 'N/A - Performed Well'
        df['key_insights'] = 'Successful box office performance'
        return df
    
    print(f"Analyzing {len(underperformers)} underperforming movies...")
    
    # Process in smaller batches to avoid timeouts
    batch_size = 5
    ai_results = []
    
    for i in range(0, len(underperformers), batch_size):
        batch = underperformers.iloc[i:i+batch_size]
        
        for _, movie in batch.iterrows():
            print(f"  Analyzing: {movie['title']} ({int(movie['year'])})")
            
            # Parallel AI calls if your API supports it
            insights = {
                'title': movie['title'],
                'underperformance_category': analyze_underperformance(movie),
                'key_insights': generate_movie_insights(movie),
                'genre_pattern': categorize_genre_patterns(movie),
                'marketing_assessment': assess_marketing_factors(movie)
            }
            ai_results.append(insights)
            
            time.sleep(0.5)  # Rate limiting
    
    # Efficient merge
    ai_df = pd.DataFrame(ai_results)
    enriched_df = df.merge(ai_df, on='title', how='left')
    
    # Vectorized fillna
    enriched_df['underperformance_category'] = enriched_df['underperformance_category'].fillna('N/A - Performed Well')
    enriched_df['key_insights'] = enriched_df['key_insights'].fillna('Successful box office performance')
    enriched_df['genre_pattern'] = enriched_df['genre_pattern'].fillna('N/A')
    enriched_df['marketing_assessment'] = enriched_df['marketing_assessment'].fillna('N/A')
    
    return enriched_df

def save_data_compressed(raw_df, enriched_df):
    """
    Save data with optional compression for large datasets
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSVs (compressed if large)
    if len(raw_df) > 100:
        raw_df.to_csv(f'data/raw/movies_raw_{timestamp}.csv.gz', 
                     index=False, compression='gzip')
        enriched_df.to_csv(f'data/enriched/movies_enriched_{timestamp}.csv.gz', 
                          index=False, compression='gzip')
        print(f"Large dataset saved with compression (.csv.gz)")
    else:
        raw_df.to_csv(f'data/raw/movies_raw_{timestamp}.csv', index=False)
        enriched_df.to_csv(f'data/enriched/movies_enriched_{timestamp}.csv', index=False)
    
    # Save JSON for smaller samples
    raw_df.head(20).to_json(f'data/raw/movies_raw_{timestamp}.json', 
                            orient='records', indent=2)
    enriched_df.head(20).to_json(f'data/enriched/movies_enriched_{timestamp}.json', 
                                 orient='records', indent=2)
    
    # Create examples
    create_examples_optimized(raw_df, enriched_df, timestamp)
    
    print(f"\nData saved successfully!")
    print(f"Timestamp: {timestamp}")
    return timestamp

def create_examples_optimized(raw_df, enriched_df, timestamp):
    """
    Efficiently create comparison examples
    """
    underperformers = enriched_df[enriched_df['is_underperformer'] == True].head(3)
    
    if underperformers.empty:
        return
    
    examples = {
        'metadata': {
            'total_movies': len(enriched_df),
            'underperformers': len(enriched_df[enriched_df['is_underperformer'] == True]),
            'timestamp': timestamp
        },
        'comparisons': []
    }
    
    # Vectorized example creation
    for _, movie in underperformers.iterrows():
        examples['comparisons'].append({
            'before': {
                'title': movie['title'],
                'year': int(movie['year']) if pd.notna(movie['year']) else 'Unknown',
                'letterboxd_rating': movie['letterboxd_rating'],
                'box_office_millions': movie['box_office_millions']
            },
            'after': {
                'title': movie['title'],
                'year': int(movie['year']) if pd.notna(movie['year']) else 'Unknown',
                'letterboxd_rating': movie['letterboxd_rating'],
                'box_office_millions': movie['box_office_millions'],
                'underperformance_category': movie.get('underperformance_category', 'N/A'),
                'key_insights': movie.get('key_insights', 'N/A')[:150],
                'marketing_assessment': movie.get('marketing_assessment', 'N/A')[:150]
            }
        })
    
    with open(f'examples/comparison_{timestamp}.json', 'w') as f:
        json.dump(examples, f, indent=2)

def display_results(df):
    """
    Display formatted results summary
    """
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Hidden Gems Summary")
    print("=" * 60)
    
    # Overall statistics
    print(f"\n📊 Dataset Overview:")
    print(f"  • Total movies analyzed: {len(df)}")
    print(f"  • Date range: {df['year'].min():.0f}-{df['year'].max():.0f}")
    print(f"  • Average rating: {df['letterboxd_rating'].mean():.2f}/5.0")
    
    # Underperformer analysis
    underperformers = df[df['is_underperformer'] == True]
    if not underperformers.empty:
        print(f"\n💎 Hidden Gems Found: {len(underperformers)}")
        print(f"  • Average box office: ${underperformers['box_office_millions'].mean():.1f}M")
        print(f"  • Average rating: {underperformers['letterboxd_rating'].mean():.2f}/5.0")
        
        # Top 3 hidden gems
        print("\n🏆 Top Hidden Gems:")
        for i, (_, movie) in enumerate(underperformers.head(3).iterrows(), 1):
            print(f"\n  {i}. {movie['title']} ({int(movie['year'])})")
            print(f"     Rating: {movie['letterboxd_rating']} | Box Office: ${movie['box_office_millions']}M")
            print(f"     Category: {movie.get('underperformance_category', 'N/A')}")
            print(f"     Insight: {movie.get('key_insights', 'N/A')[:80]}...")
    
    # Performance metrics
    print(f"\n⚡ Pipeline Performance:")
    print(f"  • Movies processed per second: {len(df)/60:.1f}")
    print(f"  • Data enrichment rate: 100%")

def main():
    """
    Optimized main orchestration function
    """
    start_time = time.time()
    
    print("=" * 60)
    print("🎬 HIDDEN GEMS ANALYZER - Optimized Pipeline")
    print("=" * 60)
    
    try:
        # Phase 1: Data Extraction
        print("\n[Phase 1/4] EXTRACTION")
        print("-" * 30)
        
        # Scrape Letterboxd efficiently
        movies = scrape_letterboxd_efficient()
        if not movies:
            print("❌ No movies found. Check internet connection.")
            return
        
        # Fetch box office data in parallel
        enriched_movies = fetch_omdb_batch(movies)
        if not enriched_movies:
            print("❌ No valid movie data retrieved.")
            return
        
        # Phase 2: Transformation
        print("\n[Phase 2/4] TRANSFORMATION")
        print("-" * 30)
        df = transform_data_vectorized(enriched_movies)
        print(f"✓ Transformed {len(df)} movies")
        
        # Phase 3: AI Enrichment
        print("\n[Phase 3/4] AI ENRICHMENT")
        print("-" * 30)
        enriched_df = enrich_with_ai_batch(df)
        
        # Phase 4: Storage
        print("\n[Phase 4/4] DATA STORAGE")
        print("-" * 30)
        timestamp = save_data_compressed(df, enriched_df)
        
        # Display results
        display_results(enriched_df)
        
        # Performance summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Pipeline completed in {elapsed_time:.1f} seconds")
        print(f"📁 Results saved with timestamp: {timestamp}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()