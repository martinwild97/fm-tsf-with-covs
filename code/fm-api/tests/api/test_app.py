import pytest
from httpx import AsyncClient, ASGITransport
from src.api.app import (
    app,
    lifespan,
)

@pytest.fixture(scope="module")
async def client():
    # Provides an AsyncClient instance with ASGITransport and FastAPI lifespan management.
    transport = ASGITransport(app=app)
    async with lifespan(app):
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client


@pytest.mark.anyio
async def test_get_model___default_state___returns_timesfm_info(client: AsyncClient):
    # Arrange - No specific arrangement needed, using default state

    # Act
    response = await client.get("/model/")

    # Assert
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_type"] == "timesfm"
    assert isinstance(data["model_path"], str) or data["model_path"] is None
    assert isinstance(data["model_kwargs"], dict)


@pytest.mark.anyio
async def test_post_model___valid_moirai_with_path___updates_successfully_and_get_reflects_change(
    client: AsyncClient,
):
    # Arrange
    payload = {"model_type": "moirai", "model_path": "Salesforce/moirai-1.1-R-small"}

    # Act - Update model
    update_response = await client.post("/model/", json=payload)

    # Assert - Update successful
    assert update_response.status_code == 200, update_response.text
    update_data = update_response.json()
    assert "Active model updated" in update_data.get("status", "")

    # Act - Verify current model info
    get_response = await client.get("/model/")

    # Assert - Verification successful
    assert get_response.status_code == 200, get_response.text
    current_model_info = get_response.json()
    assert current_model_info["model_type"] == "moirai"
    assert current_model_info["model_path"] == "Salesforce/moirai-1.1-R-small"


@pytest.mark.anyio
async def test_post_model___valid_ttm_no_path___updates_successfully_and_get_reflects_change(
    client: AsyncClient,
):
    # Arrange
    payload = {"model_type": "ttm"}

    # Act - Update model
    update_response = await client.post("/model/", json=payload)

    # Assert - Update successful
    assert update_response.status_code == 200, update_response.text
    update_data = update_response.json()
    assert "Active model updated" in update_data.get("status", "")

    # Act - Verify current model info
    get_response = await client.get("/model/")

    # Assert - Verification successful
    assert get_response.status_code == 200, get_response.text
    current_model_info = get_response.json()
    assert current_model_info["model_type"] == "ttm"
    # Check if model_path is None (if no default path is set) or a string (if a default path exists for ttm)
    assert current_model_info["model_path"] is None or isinstance(
        current_model_info["model_path"], str
    )


@pytest.mark.anyio
async def test_post_forecast___valid_input_with_active_model___returns_correct_forecast_structure(
    client: AsyncClient,
):
    # Arrange
    # Ensure a default model is active (implicitly handled by fixture scope or previous tests,
    # though ideally tests are independent. For this example, we assume the default 'timesfm' or
    # the state left by previous tests is sufficient).
    forecast_input = {
        "forecast_horizon": 4,
        "actuals": {
            "timestamp": [f"2024-0{i + 1}-01" for i in range(9)],
            "data": {"value": list(range(9))},
        },
    }

    # Act
    response = await client.post("/forecast/", json=forecast_input)

    # Assert
    assert response.status_code == 200, response.text
    forecast_data = response.json()
    assert "timestamp" in forecast_data
    assert "data" in forecast_data
    assert "lower" in forecast_data  # Assuming prediction intervals are returned
    assert "upper" in forecast_data  # Assuming prediction intervals are returned
    assert isinstance(forecast_data["data"], dict)
    assert "value" in forecast_data["data"]  # Ensure the 'value' key exists within data
    assert len(forecast_data["data"]["value"]) == 4
    assert len(forecast_data["timestamp"]) == 4
    # Check the format of the first timestamp - adapt if your API uses a different format
    assert (
        forecast_data["timestamp"][0] == "2024-10-01T00:00:00"
    )  # Adjusted based on original comment


@pytest.mark.anyio
async def test_post_model___invalid_model_type___returns_400_error(client: AsyncClient):
    # Arrange
    payload = {"model_type": "invalid_model"}

    # Act
    response = await client.post("/model/", json=payload)

    # Assert
    assert response.status_code == 400, response.text
    error_data = response.json()
    assert "detail" in error_data
    assert "Model 'invalid_model' is not implemented" in error_data["detail"]


@pytest.mark.anyio
async def test_post_forecast___no_active_model___returns_400_error(client: AsyncClient):
    # Arrange
    # Attempt to set an invalid model to ensure no valid model is active.
    # Note: This makes the test dependent on the behavior of the POST /model/ endpoint
    # with invalid data, specifically that it *unsets* the active model or leaves it unset.
    invalid_payload = {"model_type": "non_existent_model_to_clear_state"}
    await client.post(
        "/model/", json=invalid_payload
    )  # We expect this to fail (400) but potentially clear the model

    forecast_input = {
        "forecast_horizon": 4,
        "actuals": {
            "timestamp": [f"2024-0{i + 1}-01" for i in range(9)],
            "data": {"value": list(range(9))},
        },
    }

    # Act
    response = await client.post("/forecast/", json=forecast_input)

    # Assert
    assert response.status_code == 400, response.text
    error_data = response.json()
    assert "detail" in error_data
    # The exact error message might vary depending on implementation
    # Adjust "No active model loaded" if your API returns a different specific message.
    assert "No active model loaded" in error_data["detail"]
