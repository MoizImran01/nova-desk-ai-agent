# Reserve treatment time when a booking succeeds

The clinic serves one appointment at a time. It offers appointments daily from 9 am to 5 pm in Karachi time (`Asia/Karachi`). Appointments start on the hour or half-hour and must finish by closing. A booking reserves the calendar for its treatment’s full duration, across all treatments and patients. For example, a 45-minute appointment at 10:00 also occupies 10:30. The next possible start is 11:00.

Offering a slot in chat leaves it available. When two patients request overlapping times, the first successful booking reserves its interval and Nova offers the other patient fresh times. This applies even when the requested start times differ.

Only the patient who booked an appointment can cancel it. Their cancellation frees the reserved interval. When that patient reschedules, the original interval is released only when the move succeeds. If the new time is taken, the patient keeps the original appointment.
