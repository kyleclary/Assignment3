I used Claude to vibe code the whole project besides changing a few values manually like the underperformance threshold. The following are prompts that I used: "You are a software developer. Design a data pipeline that scrapes letterboxd for average review scores of movies and then checks box office performance for movies above a 4 average review score. Then use deepseek to enhance the data by categorizing movies that underperformed in the box office. Follow the rubric and repo structure provided in the directions. (I then pasted the rubric and repo structure)."

After Claude wrote the code, there was a problem with collecting data. I asked Claude to help me write a debugger and after using it, I found that the problem was because the film data was structured differently than expected. Claude then rewrote the scraper to collect the data effectively.

While the scraper worked, I kept getting movies from this year or movies that haven't been released yet. I asked Claude: "Can you rewrite main to filter for movies released before 2020 and make it so the box office earnings cannot be 0"

This prompt returned the final code for the project.