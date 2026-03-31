"""Tests for the game data files in starter/games/."""
import os
import json
import glob
import pytest

GAMES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'starter', 'games'))
REQUIRED_FIELDS = {"Name", "Platform", "Genre", "Publisher", "Description", "YearOfRelease"}


@pytest.fixture(scope="module")
def all_games():
    files = sorted(glob.glob(os.path.join(GAMES_DIR, '*.json')))
    games = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            games.append((os.path.basename(f), json.load(fp)))
    return games


class TestGameFiles:

    def test_exactly_15_game_files(self, all_games):
        assert len(all_games) == 15

    def test_files_named_001_to_015(self):
        files = sorted(glob.glob(os.path.join(GAMES_DIR, '*.json')))
        basenames = [os.path.basename(f) for f in files]
        expected = [f"{str(i).zfill(3)}.json" for i in range(1, 16)]
        assert basenames == expected

    def test_all_files_valid_json(self, all_games):
        for filename, game in all_games:
            assert isinstance(game, dict), f"{filename} should be a JSON object"

    def test_all_required_fields_present(self, all_games):
        for filename, game in all_games:
            missing = REQUIRED_FIELDS - set(game.keys())
            assert not missing, f"{filename} missing fields: {missing}"

    def test_year_of_release_is_integer(self, all_games):
        for filename, game in all_games:
            assert isinstance(game["YearOfRelease"], int), \
                f"{filename}: YearOfRelease should be int, got {type(game['YearOfRelease'])}"

    def test_all_string_fields_non_empty(self, all_games):
        string_fields = ["Name", "Platform", "Genre", "Publisher", "Description"]
        for filename, game in all_games:
            for field in string_fields:
                assert isinstance(game[field], str) and len(game[field].strip()) > 0, \
                    f"{filename}: {field} should be a non-empty string"

    def test_year_of_release_reasonable_range(self, all_games):
        for filename, game in all_games:
            year = game["YearOfRelease"]
            assert 1980 <= year <= 2025, \
                f"{filename}: YearOfRelease {year} out of expected range"

    def test_all_game_names_unique(self, all_games):
        names = [game["Name"] for _, game in all_games]
        assert len(names) == len(set(names)), "Duplicate game names found"

    def test_gran_turismo_is_001(self):
        filepath = os.path.join(GAMES_DIR, '001.json')
        with open(filepath, 'r') as f:
            game = json.load(f)
        assert game["Name"] == "Gran Turismo"
        assert game["YearOfRelease"] == 1997

    def test_halo_infinite_is_015(self):
        filepath = os.path.join(GAMES_DIR, '015.json')
        with open(filepath, 'r') as f:
            game = json.load(f)
        assert game["Name"] == "Halo Infinite"
        assert game["Platform"] == "Xbox Series X|S"
