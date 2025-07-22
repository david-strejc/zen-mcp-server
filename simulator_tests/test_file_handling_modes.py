"""
Test file handling modes (embedded, summary, reference) for MCP tools
"""

import json
import os
import tempfile
import time

from .base_test import BaseSimulatorTest


class TestFileHandlingModes(BaseSimulatorTest):
    """Test different file handling modes in MCP tools"""

    def get_test_name(self) -> str:
        return "file_handling_modes"

    def setup_test_files(self) -> dict:
        """Create test files to use in the test"""
        test_files = {}

        # Create a test Python file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""
Test module for file handling demonstration
"""

def process_data(data):
    """Process data with transformations"""
    result = []
    for item in data:
        if isinstance(item, str):
            result.append(item.upper())
        elif isinstance(item, int):
            result.append(item * 2)
        else:
            result.append(str(item))
    return result

class DataAnalyzer:
    """Analyze data patterns"""

    def __init__(self):
        self.data = []
        self.patterns = {}
    def add_data(self, value):
        """Add data point"""
        self.data.append(value)

    def find_patterns(self):
        """Find patterns in data"""
        # Count occurrences
        for item in self.data:
            key = str(type(item).__name__)
            if key not in self.patterns:
                self.patterns[key] = 0
            self.patterns[key] += 1
        return self.patterns

# Long function that needs refactoring
def analyze_complex_data(users, products, orders, config):
    """Complex analysis function with multiple responsibilities"""
    # Validate inputs
    if not users:
        raise ValueError("No users")
    if not products:
        raise ValueError("No products")

    # Process users
    active_users = [u for u in users if u.get('active')]

    # Calculate metrics
    total_revenue = 0
    user_orders = {}
    for order in orders:
        user_id = order['user_id']
        amount = order['amount']

        if user_id not in user_orders:
            user_orders[user_id] = []
        user_orders[user_id].append(order)
        total_revenue += amount
    # Find top products
    product_sales = {}
    for user_id, orders in user_orders.items():
        for order in orders:
            pid = order['product_id']
            if pid not in product_sales:
                product_sales[pid] = 0
            product_sales[pid] += order['amount']

    # Generate report
    report = {
        'active_users': len(active_users),
        'total_revenue': total_revenue,
        'top_products': sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5]
    }

    return report
'''
            )
            test_files["python_file"] = f.name

        # Create a larger text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for file handling modes.\n" * 100)
            f.write("\nThis file contains repeated content to make it larger.\n")
            f.write("It should be summarized when using summary mode.\n")
            test_files["text_file"] = f.name

        return test_files

    def cleanup_test_files(self, test_files: dict):
        """Clean up test files"""
        for file_path in test_files.values():
            try:
                os.unlink(file_path)
            except Exception:
                pass

    def run_test(self) -> bool:
        """Test file handling modes"""
        self.logger.info("Testing file handling modes...")

        # Create test files
        test_files = self.setup_test_files()

        try:
            # Test 1: Embedded mode (default)
            self.logger.info("Test 1: Testing embedded mode (default)")
            tool_input = {
                "prompt": "Analyze this Python code and explain what it does",
                "files": [test_files["python_file"]],
                "model": "flash",  # Using flash for testing
            }

            response_text, continuation_id = self.call_mcp_tool("chat", tool_input)
            if not response_text:
                self.logger.error("Embedded mode test failed: No response")
                return False

            # Parse the response
            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse response: {response_text}")
                return False

            if response.get("status") != "success":
                self.logger.error(f"Embedded mode test failed: {response}")
                return False

            # Check that we got a normal response
            content = response.get("content", "")
            if not content:
                self.logger.error("No content in embedded mode response")
                return False

            # Test 2: Summary mode
            self.logger.info("Test 2: Testing summary mode")
            tool_input = {
                "prompt": "Analyze these files briefly",
                "files": [test_files["python_file"], test_files["text_file"]],
                "file_handling_mode": "summary",
                "model": "flash",
            }

            response_text, continuation_id = self.call_mcp_tool("chat", tool_input)
            if not response_text:
                self.logger.error("Summary mode test failed: No response")
                return False

            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse response: {response_text}")
                return False

            if response.get("status") != "success":
                self.logger.error(f"Summary mode test failed: {response}")
                return False

            # Check for file references in response
            file_refs = response.get("file_references")
            if not file_refs:
                self.logger.error("No file references in summary mode response")
                return False

            self.logger.info(f"Got {len(file_refs)} file references in summary mode")

            # Test 3: Reference mode
            self.logger.info("Test 3: Testing reference mode")
            tool_input = {
                "prompt": "Check these files for code quality issues",
                "files": [test_files["python_file"]],
                "file_handling_mode": "reference",
                "refactor_type": "codesmells",
                "model": "flash",
            }

            response_text, continuation_id = self.call_mcp_tool("refactor", tool_input)
            if not response_text:
                self.logger.error("Reference mode test failed: No response")
                return False

            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse response: {response_text}")
                return False

            if response.get("status") != "success":
                self.logger.error(f"Reference mode test failed: {response}")
                return False

            # Check for file references
            file_refs = response.get("file_references")
            if not file_refs:
                self.logger.error("No file references in reference mode response")
                return False

            # Test 4: Retrieve stored file
            self.logger.info("Test 4: Testing file retrieval")
            if file_refs:
                ref_id = file_refs[0].get("reference_id")
                if ref_id:
                    retrieve_input = {"reference_id": ref_id, "model": "flash"}

                    retrieve_text, _ = self.call_mcp_tool("fileretrieve", retrieve_input)
                    if not retrieve_text:
                        self.logger.error("File retrieve failed: No response")
                        return False

                    try:
                        retrieve_response = json.loads(retrieve_text)
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse retrieve response: {retrieve_text}")
                        return False

                    if retrieve_response.get("status") != "success":
                        self.logger.error(f"File retrieve failed: {retrieve_response}")
                        return False

                    # Check that we got the file content back
                    content = retrieve_response.get("content", "")
                    if "process_data" not in content:
                        self.logger.error("Retrieved file doesn't contain expected content")
                        return False

                    self.logger.info("Successfully retrieved stored file")

            # Test 5: Cross-tool continuation with file handling
            self.logger.info("Test 5: Testing cross-tool continuation with files")

            # First call with reference mode
            tool_input = {
                "prompt": "Analyze this code structure",
                "files": [test_files["python_file"]],
                "file_handling_mode": "reference",
                "model": "flash",
            }

            response_text, continuation_id = self.call_mcp_tool("analyze", tool_input)
            if not response_text:
                self.logger.error("Analyze with reference mode failed: No response")
                return False

            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse response: {response_text}")
                return False

            if response.get("status") != "success":
                self.logger.error(f"Analyze with reference mode failed: {response}")
                return False

            thread_id = response.get("conversation", {}).get("thread_id")
            if not thread_id:
                self.logger.error("No thread ID in analyze response")
                return False

            # Continue with different tool
            followup_input = {
                "prompt": "Now help me refactor the complex function",
                "continuation_id": thread_id,
                "refactor_type": "decompose",
                "model": "flash",
            }

            followup_text, _ = self.call_mcp_tool("refactor", followup_input)
            if not followup_text:
                self.logger.error("Continuation with refactor failed: No response")
                return False

            try:
                followup_response = json.loads(followup_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse followup response: {followup_text}")
                return False

            if followup_response.get("status") != "success":
                self.logger.error(f"Continuation with refactor failed: {followup_response}")
                return False

            self.logger.info("Cross-tool continuation with file handling successful")

            # Validate through logs
            time.sleep(2)  # Wait for logs

            logs = self.get_docker_logs(lines=100)
            if "file_handling_mode" not in logs:
                self.logger.error("File handling mode not found in logs")
                return False

            if "Storing file with reference ID" not in logs:
                self.logger.warning("File storage log not found (might be using embedded mode)")

            self.logger.info("✅ All file handling mode tests passed!")
            return True

        finally:
            # Cleanup test files
            self.cleanup_test_files(test_files)
