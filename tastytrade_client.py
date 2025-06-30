import requests
import json
import streamlit as st
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TastyTradeClient:
    """TastyTrade API client with OAuth2 authentication"""
    
    def __init__(self):
        self.base_url = "https://api.tastytrade.com"
        self.session_token = None
        self.remember_token = None
        self.expires_at = None
        
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate using username/password"""
        try:
            auth_url = f"{self.base_url}/sessions"
            
            payload = {
                "login": username,
                "password": password,
                "remember-me": True
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(auth_url, json=payload, headers=headers)
            
            if response.status_code == 201:
                data = response.json()
                self.session_token = data['data']['session-token']
                self.remember_token = data['data'].get('remember-token')
                self.expires_at = datetime.now() + timedelta(hours=24)
                
                logger.info("TastyTrade authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token"""
        if not self.session_token:
            raise ValueError("Not authenticated")
        
        return {
            "Authorization": f"{self.session_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def is_authenticated(self) -> bool:
        """Check if authenticated and not expired"""
        return (
            self.session_token is not None and 
            self.expires_at is not None and 
            datetime.now() < self.expires_at
        )
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        if not self.is_authenticated():
            return None
        
        try:
            url = f"{self.base_url}/instruments/equities/{symbol}"
            response = requests.get(url, headers=self.get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return data['data'].get('market-price')
            return None
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_options_chain(self, symbol: str) -> Optional[Dict]:
        """Get options chain for a symbol"""
        if not self.is_authenticated():
            return None
        
        try:
            url = f"{self.base_url}/option-chains/{symbol}/nested"
            response = requests.get(url, headers=self.get_headers())
            
            if response.status_code == 200:
                return response.json()
            return None
                
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return None
