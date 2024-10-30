def compare_solutions(solution1, solution2):
    """Compare two solutions for similarity"""
    try:
        from difflib import SequenceMatcher
        
        # Extract text from solution_text lists
        text1 = ' '.join([item['text'] for item in solution1['solution_text']])
        text2 = ' '.join([item['text'] for item in solution2['solution_text']])
        
        # Compare text similarity
        text_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Extract LaTeX from equations lists
        latex1 = ' '.join([eq['latex'] for eq in solution1['equations']])
        latex2 = ' '.join([eq['latex'] for eq in solution2['equations']])
        
        # Compare equation similarity
        equation_similarity = SequenceMatcher(None, latex1, latex2).ratio()
        
        return {
            'text_similarity': text_similarity,
            'equation_similarity': equation_similarity,
            'overall_similarity': (text_similarity + equation_similarity) / 2
        }
    except Exception as e:
        raise Exception(f"Error comparing solutions: {str(e)}")