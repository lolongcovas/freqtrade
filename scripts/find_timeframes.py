import copy
import subprocess
import json


base_config = {
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "max_open_trades": 10,
    "stake_currency": "USDT",
    "stake_amount": 50,
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "timeframe": "5m",
    "dry_run": True,
    "cancel_open_orders_on_exit": False,
    "minimal_roi": {
        "0": 0.10
    },
    "unfilledtimeout": {
        "entry": 15,
        "exit": 30,
        "unit": "minutes"
    },
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": True,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": False,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing": {
        "price_side": "other",
        "use_order_book": True,
        "order_book_top": 1
    },
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "ccxt_config": {
            "enableRateLimit": True
        },
        "ccxt_async_config": {
            "enableRateLimit": True,
            "rateLimit": 200
        },
        "pair_whitelist": [
            "BTC/USDT"
        ],
        "pair_blacklist": [
        ]
    },
    "pairlists": [
        {
            "method": "StaticPairList"
        }
    ],
    "edge": {
        "enabled": False,
        "process_throttle_secs": 3600,
        "calculate_since_number_of_days": 7,
        "allowed_risk": 0.01,
        "stoploss_range_min": -0.01,
        "stoploss_range_max": -0.1,
        "stoploss_range_step": -0.01,
        "minimum_winrate": 0.60,
        "minimum_expectancy": 0.20,
        "min_trade_number": 10,
        "max_trade_duration_minute": 1440,
        "remove_pumps": False
    },
    "telegram": {
        "enabled": False,
        "token": "",
        "chat_id": ""
    },
    "api_server": {
        "enabled": False,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8082,
        "verbosity": "error",
        "enable_openapi": False,
        "jwt_secret_key": "95946303c04fd27d914b27f3324859bf0d50e6f75eace22fa7834dbb8aa38a85",
        "CORS_origins": [],
        "username": "btc",
        "password": "tothemoon"
    },
    "bot_name": "freqtrade",
    "initial_state": "running",
    "force_entry_enable": False,
    "internals": {
        "process_throttle_secs": 5
    }
}


config_ai = {
        "freqai": {
            "enabled": True,
            "keras": False,
            "startup_candles": 3000,
            "purge_old_models": True,
            "train_period_days": 10,
            "backtest_period_days": 0.05,
            "live_retrain_hours": 0,
            "identifier": "finetuning-timeframe-model",
            "expiration_hours": 2,
            "live_trained_timestamp": 0,
            "feature_parameters": {
                "include_timeframes": [ # we fine tune this parameter
                    
                ],
                "include_corr_pairlist": [
                    "BTC/USDT",
                    "ETH/USDT"
                ],
                "label_period_candles": 12,
                "include_shifted_candles": 2,
                "DI_threshold": 0.9,
                "weight_factor": 0.99,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "use_DBSCAN_to_remove_outliers": True,
                "svm_nu": 0.1,
                "stratify_training_data": 0,
                "indicator_max_period_candles": 500,
                "indicator_periods_candles": [14]
            },
            "data_split_parameters": {
                "test_size": 0.1,
                "random_state": 1,
                "shuffle": False
            },
            "model_training_parameters": {
                "n_estimators": 600,
                "task_type": "CPU",
                "learning_rate": 0.02,
                "thread_count": 6,
                "model_size_reg": 2
            }
        }
    }

config_use = "/tmp/config.json"

param = 'include_timeframes'

for tf in ["5m", "15m", "30m", "1h", "2h"]:

    config = copy.deepcopy(base_config)
    config["freqai"] = copy.deepcopy(config_ai)["freqai"]
    exp_id = "_".join([tf])
    config['freqai']['feature_parameters'].update({"include_timeframes": [tf]})
    config['freqai'].update({"identifier": f'test-{param}-{exp_id}'})
    config_run = f'{config_use}_{exp_id}.json'
    with open(config_run, 'w') as fid:
        json.dump(config, fid)

    subprocess.call([
            #"python3.9", "-m", "freqtrade",
            "VENV/bin/python3.8", "freqtrade/main.py",
            "backtesting",
            "--config", config_run,
            #"--datadir", "user_data/data/binance/hyperparameters-search",
            "--freqaimodel", "CatboostPredictionBinaryMultiModel",
            "--freqaimodel-path", "user_data/freqaimodels/",
            "--strategy", "FreqaiBinaryClassStrategy",
            "--strategy-path", "user_data/strategies",
            "--timerange", "20220715-20220815",
            "--export", "signals",
            "--cache", "none",
            f"--export-filename=user_data/backtest_results/order_{exp_id}.json"
    ])