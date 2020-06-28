from app import app
import unittest

class EmotionTestCase(unittest.TestCase):

    # Ensure that Flask was set up correctly
	def test_index(self):
		tester = app.test_client(self)
		response = tester.get('/', content_type='html/text')
		self.assertEqual(response.status_code, 200)

	def test_video(self):
		tester = app.test_client(self)
		response = tester.get('video_feed', content_type='html/text')
		self.assertEqual(response.status_code, 200)

	def test_heading(self):
		tester = app.test_client(self)
		response = tester.get('/', content_type='html/text')
		self.assertTrue(b'Real-Time Emotion Recognition System' in response.data)

	def test_subHeading(self):
		tester = app.test_client(self)
		response = tester.get('/', content_type='html/text')
		self.assertTrue(b'Here We go !' in response.data)

	def test_button(self):
		tester = app.test_client(self)
		response = tester.get('/', content_type='html/text')
		self.assertTrue(b'Identify Me' in response.data)

if __name__ == '__main__':
	unittest.main()
