import re
import json
import os

class DataExtractor:
    def __init__(self, patterns_config_path=None):
        self.patterns_config_path = patterns_config_path
        self.patterns = self._load_patterns() if patterns_config_path else self._default_patterns()

    def _load_patterns(self):
        try:
            with open(self.patterns_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading patterns config: {e}")
            return self._default_patterns()

    def _default_patterns(self):
        return {
            'routing_number': r'\b([0-9]{9})\b',
            'account_number': r'\b([0-9]{10,14})\b',
            'check_number': r'\b([0-9]{3,6})\b',
            'date': r'\b(0[1-9]|1[0-2])[-/\.](0[1-9]|[12][0-9]|3[01])[-/\.]((19|20)\d{2}|\d{2})\b',
            'amount': r'\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}|[0-9]+\.[0-9]{2})',
            'written_amount': r'(?:(?:pay to the order of|pay).{0,50})?((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and|-|\s|dollars|cents)+)'
        }

    def extract_routing_number(self, micr_text):
        pattern = self.patterns['routing_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""

    def extract_account_number(self, micr_text):
        pattern = self.patterns['account_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""

    def extract_check_number(self, micr_text):
        pattern = self.patterns['check_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""

    def extract_date(self, date_text):
        pattern = self.patterns['date']
        match = re.search(pattern, date_text)
        if match:
            month, day, year = match.groups()[0], match.groups()[1], match.groups()[2]
            year = f"20{year}" if len(year) == 2 and int(year) <= 99 else year
            return f"{month}/{day}/{year}"
        return ""

    def extract_amount(self, amount_text):
        clean_text = amount_text.replace(' ', '')
        pattern = self.patterns['amount']
        match = re.search(pattern, clean_text)
        return match.group(1).replace(',', '') if match else ""

    def extract_payee(self, payee_text):
        payee_pattern = r'(?:pay to the order of|pay to|payto)[ :]*([^\n\r$]+)'
        match = re.search(payee_pattern, payee_text.lower())
        if match:
            payee = match.group(1).strip()
            payee = re.sub(r'(?:dollars|and \d+/100|\d+/100).*$', '', payee, flags=re.IGNORECASE).strip()
            return payee
        return payee_text.strip()

    def extract_written_amount(self, written_amount_text):
        pattern = self.patterns['written_amount']
        match = re.search(pattern, written_amount_text.lower())
        if match:
            written_amount = match.group(1).strip()
            written_amount = re.sub(r'(dollars|only)$', '', written_amount, flags=re.IGNORECASE).strip()
            written_amount = re.sub(r'\s+', ' ', written_amount)
            return written_amount
        return ""