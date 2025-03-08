import re
import json
import os

class DataExtractor:
    """
    Class for extracting structured data from OCR text using pattern matching.
    """
    
    def __init__(self, patterns_config_path=None):
        """
        Initialize the data extractor.
        
        Args:
            patterns_config_path (str, optional): Path to patterns config file.
        """
        self.patterns_config_path = patterns_config_path
        self.patterns = self._load_patterns() if patterns_config_path else self._default_patterns()
    
    def _load_patterns(self):
        """
        Load extraction patterns from JSON config file.
        
        Returns:
            dict: Dictionary of regex patterns.
        """
        try:
            with open(self.patterns_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading patterns config: {e}")
            return self._default_patterns()
    
    def _default_patterns(self):
        """
        Provide default extraction patterns.
        
        Returns:
            dict: Dictionary of regex patterns.
        """
        return {
            'routing_number': r'\b([0-9]{9})\b',
            'account_number': r'\b([0-9]{10,12})\b',
            'check_number': r'\b([0-9]{4})\b',
            'date': r'\b(0[1-9]|1[0-2])[/.-](0[1-9]|[12][0-9]|3[01])[/.-](19|20)\d\d\b',
            'amount': r'\$?\s*([0-9,]+\.[0-9]{2})',
            'written_amount': r'(?:(?:pay to the order of|pay).{1,50})?((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|and|\s|dollars|cents)+)'
        }
    
    def extract_routing_number(self, micr_text):
        """
        Extract routing number from MICR line text.
        
        Args:
            micr_text (str): MICR line text.
            
        Returns:
            str: Extracted routing number or empty string if not found.
        """
        pattern = self.patterns['routing_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""
    
    def extract_account_number(self, micr_text):
        """
        Extract account number from MICR line text.
        
        Args:
            micr_text (str): MICR line text.
            
        Returns:
            str: Extracted account number or empty string if not found.
        """
        pattern = self.patterns['account_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""
    
    def extract_check_number(self, micr_text):
        """
        Extract check number from MICR line text.
        
        Args:
            micr_text (str): MICR line text.
            
        Returns:
            str: Extracted check number or empty string if not found.
        """
        pattern = self.patterns['check_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""
    
    def extract_date(self, date_text):
        """
        Extract date from date region text.
        
        Args:
            date_text (str): Date region text.
            
        Returns:
            str: Extracted date or empty string if not found.
        """
        pattern = self.patterns['date']
        match = re.search(pattern, date_text)
        return match.group(0) if match else ""
    
    def extract_amount(self, amount_text):
        """
        Extract numeric amount from amount region text.
        
        Args:
            amount_text (str): Amount region text.
            
        Returns:
            str: Extracted amount or empty string if not found.
        """
        # Clean up text - remove spaces and non-relevant characters
        clean_text = amount_text.replace(' ', '')
        
        pattern = self.patterns['amount']
        match = re.search(pattern, clean_text)
        
        if match:
            # Remove commas and return clean amount
            return match.group(1).replace(',', '')
        else:
            # Try alternative pattern for amounts without decimal point
            alt_pattern = r'\$?\s*([0-9,]+)'
            alt_match = re.search(alt_pattern, clean_text)
            if alt_match:
                # Add decimal .00 and remove commas
                return alt_match.group(1).replace(',', '') + '.00'
            else:
                return ""
    
    def extract_payee(self, payee_text):
        """
        Extract payee name from payee line text.
        
        Args:
            payee_text (str): Payee line text.
            
        Returns:
            str: Extracted payee name or empty string if not found.
        """
        # Look for patterns like "PAY TO THE ORDER OF" or similar
        payee_pattern = r'(?:pay to the order of|pay to|payto)[ :]*([^\n\r$]+)'
        match = re.search(payee_pattern, payee_text.lower())
        
        if match:
            payee = match.group(1).strip()
            # Remove any trailing text that might be part of amount
            payee = re.sub(r'(?:dollars|and \d+/100|\d+/100).*$', '', payee, flags=re.IGNORECASE).strip()
            return payee
        else:
            # If no pattern matches, return the whole text as it might be just the payee name
            return payee_text.strip()
    
    def extract_written_amount(self, written_amount_text):
        """
        Extract written amount from text.
        
        Args:
            written_amount_text (str): Text containing the written amount.
            
        Returns:
            str: Extracted written amount or empty string if not found.
        """
        pattern = self.patterns['written_amount']
        match = re.search(pattern, written_amount_text.lower())
        
        if match:
            # Extract the written amount and clean it up
            written_amount = match.group(1).strip()
            # Remove any trailing "dollars" or "only" words
            written_amount = re.sub(r'(dollars|only)$', '', written_amount, flags=re.IGNORECASE).strip()
            # Clean up extra spaces
            written_amount = re.sub(r'\s+', ' ', written_amount)
            return written_amount
        else:
            return ""
        
      