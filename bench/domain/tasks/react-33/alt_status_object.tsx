import * as React from 'react';
import * as ReactDOM from 'react-dom';

// Keeps the whole status object and derives "busy" from the submitted
// FormData (non-null exactly while the form is pending), and builds the
// button with createElement instead of JSX.
export const SubmitButton = ({
  children,
  pendingText,
}: {
  children: React.ReactNode;
  pendingText: string;
}): React.JSX.Element => {
  const status = ReactDOM.useFormStatus();
  const busy = status.data !== null;
  const label = busy ? <span>{pendingText}</span> : <>{children}</>;
  return React.createElement('button', { type: 'submit', disabled: busy }, label);
};
