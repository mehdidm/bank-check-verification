import re
import logging
from rapidfuzz import fuzz


class CheckValidator:
    """
    Comprehensive validation for bank check details
    Provides multiple validation checks and rules
    """

    def __init__(self):
        """
        Initialize check validation rules and logging
        """
        self.logger = logging.getLogger(__name__)

        self.validation_rules = {
            'amount_format': r'^\d+([.,]\d{1,2})?$',
            'min_amount': 0,
            'max_amount': 1000000,
            'bank_name_min_length': 2,
            'amount_similarity_threshold': 70
        }

    def validate_amount(self, numeric_amount, letter_amount):
        """
        Validate check amount with multiple criteria

        Args:
            numeric_amount (str): Amount in numeric form
            letter_amount (str): Amount in letter form

        Returns:
            tuple: (is_valid, validation_message)
        """
        try:
            # Validate numeric amount format
            if not re.match(self.validation_rules['amount_format'], str(numeric_amount)):
                return False, "Invalid numeric amount format"

            # Convert to float
            try:
                amount = float(str(numeric_amount).replace(',', '.'))
            except ValueError:
                return False, "Unable to convert amount"

            # Check amount range
            if (amount < self.validation_rules['min_amount'] or
                    amount > self.validation_rules['max_amount']):
                return False, "Amount out of valid range"

            # Compare numeric and letter amounts using fuzzy matching
            similarity = fuzz.ratio(str(numeric_amount), str(letter_amount))
            if similarity < self.validation_rules['amount_similarity_threshold']:
                return False, "Numeric and letter amounts do not match"

            return True, "Amount validated successfully"

        except Exception as e:
            self.logger.error(f"Amount validation error: {e}")
            return False, "Validation process failed"

    def validate_bank_name(self, bank_name):
        """
        Validate bank name with multiple checks

        Args:
            bank_name (str): Bank name to validate

        Returns:
            tuple: (is_valid, validation_message)
        """
        try:
            # Check minimum length
            if len(bank_name) < self.validation_rules['bank_name_min_length']:
                return False, "Bank name too short"

            # Check for numeric characters
            if any(char.isdigit() for char in bank_name):
                return False, "Bank name should not contain numbers"

            # Optional advanced validation
            banned_words = ['fake', 'test', 'dummy']
            if any(word in bank_name.lower() for word in banned_words):
                return False, "Suspicious bank name detected"

            return True, "Bank name validated successfully"

        except Exception as e:
            self.logger.error(f"Bank name validation error: {e}")
            return False, "Validation process failed"