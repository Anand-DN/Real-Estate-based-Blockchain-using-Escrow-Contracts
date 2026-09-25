import { render, screen } from '@testing-library/react';
import App from './App';

test('renders Millow brand', () => {
  render(<App />);
  const brandElement = screen.getByRole('heading', { name: 'Millow' });
  expect(brandElement).toBeInTheDocument();
});
