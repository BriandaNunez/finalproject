import pytest
import logging

#from finalproject_itesm_mlops import cli

# Configurar el logger
logging.basicConfig(filename= r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\tests\test_finalproject_itesm_mlops.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def test_something():
    """Test something."""
    # Agregar código de prueba aquí
    logger.info("Running test_something")
    assert True


def test_command_line_interface():
    """Test the CLI."""
    result = cli.main()
    assert result.exit_code == 0
    assert '--help' in result.output
    logger.info("Running test_command_line_interface")


if __name__ == "__main__":
    pytest.main()
