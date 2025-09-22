"""
DeepSeek AI Enrichment Module
Provides intelligent analysis and categorization of movie performance data
"""

import requests
import json
import time
from typing import Dict, Any, List

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def make_deepseek_request(prompt: str, max_tokens: int = 150) -> str:
    """
    Generic function to make requests to DeepSeek API
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a film industry analyst specializing in box office performance and audience reception."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API error: {e}")
        return "Analysis unavailable due to API error"
    except KeyError:
        return "Unexpected response format from DeepSeek API"

def analyze_underperformance(movie: Dict[str, Any]) -> str:
    """
    Categorizes why a highly-rated movie underperformed at the box office
    
    Categories:
    - Limited Release: Art house or festival circuit only
    - Poor Marketing: Inadequate promotion despite quality
    - Niche Appeal: Too specialized for mainstream audiences
    - Bad Timing: Released against blockbusters or wrong season
    - Genre Barrier: Difficult genres for mainstream (foreign, documentary, experimental)
    - Distribution Issues: Problems with theater availability
    - Word-of-Mouth Failure: Couldn't build momentum despite quality
    """
    
    prompt = f"""Analyze why this critically acclaimed movie underperformed at the box office.

Movie: {movie.get('title', 'Unknown')} ({movie.get('year', 'N/A')})
Letterboxd Rating: {movie.get('letterboxd_rating', 0)}/5 (Excellent)
Box Office: ${movie.get('box_office_millions', 0)}M (Poor)
Genre: {movie.get('genre', 'Unknown')}
Director: {movie.get('director', 'Unknown')}
Plot: {movie.get('plot', 'N/A')[:200]}

Based on these factors, categorize the main reason for underperformance into EXACTLY ONE of these categories:
1. Limited Release
2. Poor Marketing
3. Niche Appeal
4. Bad Timing
5. Genre Barrier
6. Distribution Issues
7. Word-of-Mouth Failure

Respond with ONLY the category name, nothing else."""

    category = make_deepseek_request(prompt, max_tokens=20)
    
    # Validate response is one of our categories
    valid_categories = [
        "Limited Release", "Poor Marketing", "Niche Appeal", 
        "Bad Timing", "Genre Barrier", "Distribution Issues", 
        "Word-of-Mouth Failure"
    ]
    
    if category not in valid_categories:
        # If response doesn't match, try to map it
        if any(cat.lower() in category.lower() for cat in valid_categories):
            for cat in valid_categories:
                if cat.lower() in category.lower():
                    return cat
        return "Niche Appeal"  # Default fallback
    
    return category

def generate_movie_insights(movie: Dict[str, Any]) -> str:
    """
    Generates key insights about why the movie is a 'hidden gem'
    """
    
    prompt = f"""Provide a concise insight about why this movie is a "hidden gem" - highly rated but commercially unsuccessful.

Movie: {movie.get('title', 'Unknown')}
Year: {movie.get('year', 'N/A')}
Letterboxd Rating: {movie.get('letterboxd_rating', 0)}/5
IMDB Rating: {movie.get('imdb_rating', 0)}/10
Box Office: ${movie.get('box_office_millions', 0)}M
Genre: {movie.get('genre', 'Unknown')}
Awards: {movie.get('awards', 'N/A')}
Plot: {movie.get('plot', 'N/A')[:150]}

In 2-3 sentences, explain what makes this film special despite its commercial failure. Focus on artistic merit, cultural impact, or unique qualities."""

    return make_deepseek_request(prompt, max_tokens=100)

def categorize_genre_patterns(movie: Dict[str, Any]) -> str:
    """
    Identifies patterns in genre that might explain performance
    """
    
    prompt = f"""Analyze the genre pattern for box office performance.

Movie Genre: {movie.get('genre', 'Unknown')}
Box Office: ${movie.get('box_office_millions', 0)}M
Rating: {movie.get('letterboxd_rating', 0)}/5

Categorize the genre's commercial viability as one of:
- Mainstream Friendly (Action, Comedy, Thriller)
- Prestige Picture (Drama, Biography, Historical)
- Arthouse Favorite (Experimental, Foreign, Independent)
- Cult Potential (Horror, Sci-Fi, Fantasy with unique vision)
- Documentary/Educational

Respond with ONLY the category, nothing else."""

    response = make_deepseek_request(prompt, max_tokens=20)
    
    # Clean up response
    valid_patterns = [
        "Mainstream Friendly",
        "Prestige Picture",
        "Arthouse Favorite",
        "Cult Potential",
        "Documentary/Educational"
    ]
    
    for pattern in valid_patterns:
        if pattern.lower() in response.lower():
            return pattern
    
    return "Arthouse Favorite"  # Default for highly rated underperformers

def assess_marketing_factors(movie: Dict[str, Any]) -> str:
    """
    Assesses potential marketing and distribution factors
    """
    
    prompt = f"""Based on this movie's profile, assess its likely marketing challenge.

Title: {movie.get('title', 'Unknown')}
Year: {movie.get('year', 'N/A')}
Country: {movie.get('country', 'Unknown')}
Language: {movie.get('language', 'Unknown')}
Runtime: {movie.get('runtime', 'N/A')}
Actors: {movie.get('actors', 'Unknown')[:100]}

Identify the PRIMARY marketing challenge in 1-2 sentences. Consider factors like:
- Star power (or lack thereof)
- Language barriers
- Runtime issues
- Release strategy
- Target audience identification"""

    return make_deepseek_request(prompt, max_tokens=80)

def batch_analyze_trends(movies_df) -> Dict[str, Any]:
    """
    Analyzes trends across all underperforming movies
    """
    
    underperformers = movies_df[movies_df['is_underperformer'] == True]
    
    if len(underperformers) == 0:
        return {
            'common_patterns': 'No underperformers to analyze',
            'recommendations': 'N/A'
        }
    
    # Prepare summary data
    genres = underperformers['genre'].value_counts().head(5).to_dict()
    avg_rating = underperformers['letterboxd_rating'].mean()
    total_movies = len(underperformers)
    
    prompt = f"""Analyze patterns in these {total_movies} critically acclaimed but commercially unsuccessful films.

Average Letterboxd Rating: {avg_rating:.2f}/5
Top Genres: {json.dumps(genres, indent=2)}
All movies earned less than $50M at box office despite ratings above 4.0/5.

Provide:
1. Two key patterns you observe
2. One actionable recommendation for film distributors

Be concise and specific."""

    analysis = make_deepseek_request(prompt, max_tokens=150)
    
    return {
        'common_patterns': analysis,
        'total_analyzed': total_movies,
        'average_rating': avg_rating
    }

def generate_audience_mismatch_score(movie: Dict[str, Any]) -> float:
    """
    Generates a score (0-10) indicating the mismatch between critical and commercial success
    """
    
    prompt = f"""Rate the audience-critic divide for this movie on a scale of 0-10.

Movie: {movie.get('title', 'Unknown')}
Letterboxd Rating: {movie.get('letterboxd_rating', 0)}/5 (Critics love it)
Box Office: ${movie.get('box_office_millions', 0)}M (Audiences didn't show up)
Genre: {movie.get('genre', 'Unknown')}

0 = No divide (both critics and audiences aligned)
10 = Maximum divide (critics loved it, audiences completely ignored it)

Consider the genre expectations and typical audience for such films.
Respond with ONLY a number between 0 and 10, nothing else."""

    response = make_deepseek_request(prompt, max_tokens=10)
    
    try:
        score = float(response.strip())
        return min(max(score, 0), 10)  # Ensure it's between 0-10
    except:
        # Calculate a simple heuristic if API fails
        rating = movie.get('letterboxd_rating', 0)
        box_office = movie.get('box_office_millions', 0)
        
        if rating >= 4.5 and box_office < 10:
            return 9.0
        elif rating >= 4.0 and box_office < 30:
            return 7.0
        else:
            return 5.0

def create_recommendation_engine(movie: Dict[str, Any]) -> str:
    """
    Creates personalized recommendations for who would enjoy this hidden gem
    """
    
    prompt = f"""Create a brief audience recommendation for this hidden gem.

Movie: {movie.get('title', 'Unknown')}
Genre: {movie.get('genre', 'Unknown')}
Plot: {movie.get('plot', 'N/A')[:150]}
Rating: {movie.get('letterboxd_rating', 0)}/5

In one sentence, describe the ideal viewer for this film (e.g., "Perfect for fans of slow-burn psychological thrillers who appreciate atmospheric storytelling").

Be specific and helpful."""

    return make_deepseek_request(prompt, max_tokens=50)

def identify_revival_potential(movie: Dict[str, Any]) -> str:
    """
    Assesses whether this movie could find new life through streaming or re-release
    """
    
    current_year = 2024
    movie_year = int(movie.get('year', 2020)) if movie.get('year', '').isdigit() else 2020
    years_old = current_year - movie_year
    
    prompt = f"""Assess this hidden gem's potential for revival or rediscovery.

Movie: {movie.get('title', 'Unknown')} ({movie.get('year', 'N/A')})
Years since release: {years_old}
Genre: {movie.get('genre', 'Unknown')}
Current cultural relevance of themes: {movie.get('plot', 'N/A')[:100]}

Rate revival potential as one of:
- High: Perfect for streaming/revival
- Medium: Could find niche audience
- Low: Likely to remain obscure

Respond with ONLY the rating (High/Medium/Low), nothing else."""

    response = make_deepseek_request(prompt, max_tokens=10)
    
    if "high" in response.lower():
        return "High"
    elif "medium" in response.lower():
        return "Medium"
    elif "low" in response.lower():
        return "Low"
    else:
        # Fallback logic
        if years_old <= 5 and movie.get('letterboxd_rating', 0) >= 4.3:
            return "High"
        elif movie.get('letterboxd_rating', 0) >= 4.0:
            return "Medium"
        else:
            return "Low"

def analyze_director_factor(movie: Dict[str, Any]) -> bool:
    """
    Determines if director's reputation might have affected box office
    """
    prompt = f"""Is {movie.get('director', 'Unknown')} a well-known director to mainstream audiences?

Answer with just YES or NO."""

    response = make_deepseek_request(prompt, max_tokens=10)
    return "yes" in response.lower()

def detect_festival_darling(movie: Dict[str, Any]) -> bool:
    """
    Identifies if this was primarily a festival circuit success
    """
    awards = movie.get('awards', '').lower()
    
    festival_keywords = ['cannes', 'sundance', 'venice', 'berlin', 'toronto', 'festival', 'jury', 'palm']
    
    if any(keyword in awards for keyword in festival_keywords):
        return True
    
    # Ask AI for additional context
    prompt = f"""Based on this movie's profile, was it likely a "festival darling" that didn't get wide release?

Movie: {movie.get('title', 'Unknown')}
Awards: {movie.get('awards', 'N/A')[:100]}
Genre: {movie.get('genre', 'Unknown')}

Answer YES if this seems like a festival favorite, NO if it seems like it had normal distribution. Answer with just YES or NO."""

    response = make_deepseek_request(prompt, max_tokens=10)
    return "yes" in response.lower()

def calculate_cultural_impact_score(movie: Dict[str, Any]) -> float:
    """
    Estimates the cultural impact despite commercial failure
    """
    prompt = f"""Rate the cultural impact of this commercially unsuccessful but critically acclaimed film.

Movie: {movie.get('title', 'Unknown')}
Year: {movie.get('year', 'N/A')}
Rating: {movie.get('letterboxd_rating', 0)}/5
Awards: {movie.get('awards', 'N/A')[:100]}

On a scale of 1-10, how significant is this film's cultural impact or influence on cinema?
1 = No lasting impact
10 = Extremely influential despite box office

Respond with ONLY a number between 1 and 10."""

    response = make_deepseek_request(prompt, max_tokens=10)
    
    try:
        score = float(response.strip())
        return min(max(score, 1), 10)
    except:
        # Fallback based on ratings and awards
        if "won" in movie.get('awards', '').lower() or "oscar" in movie.get('awards', '').lower():
            return 7.0
        elif movie.get('letterboxd_rating', 0) >= 4.5:
            return 6.0
        else:
            return 4.0