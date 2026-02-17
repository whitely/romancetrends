{ pkgs, lib, config, inputs, ... }:

{
  env.PROJECT_NAME = "romancetrends";
  dotenv.enable = true;

  packages = [
    pkgs.git
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  # Python with data analysis packages
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv.enable = true;
    venv.requirements = ''
      # API clients
      praw              # Reddit API
      pytrends          # Google Trends
      google-api-python-client  # Google Books API

      # Data analysis
      pandas
      numpy
      matplotlib
      seaborn

      # ML/NLP
      scikit-learn
      textblob

      # Utilities
      python-dotenv
      requests
      tqdm
    '';
  };

  scripts.trends.exec = ''
    python src/google_trends.py "$@"
  '';

  scripts.reddit.exec = ''
    python src/reddit_analysis.py "$@"
  '';

  scripts.analyze.exec = ''
    python src/trend_scorer.py "$@"
  '';

  enterShell = ''
    echo "Romance Trends Analysis Environment"
    echo "Commands: trends, reddit, analyze"
  '';
}
